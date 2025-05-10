# dlc_manager.py
import deeplabcut
import pandas as pd
import cv2
import yaml
import math
import shutil
from pathlib import Path
import time
from tqdm import tqdm
import h5py, numpy as np, matplotlib.pyplot as plt
import base64, glob
from fpdf import FPDF
from xhtml2pdf import pisa
from matplotlib.backends.backend_pdf import PdfPages
import re
import textwrap
from scipy.signal import find_peaks, peak_widths

def get_project_paths(path):
    path = Path(path)
    if path.name == "config.yaml":
        config = path
        project = path.parent
    else:
        project = path
        config = project / "config.yaml"

    if not config.exists():
        raise FileNotFoundError(f"config.yaml not found at {config}")
    
    video_dir = project / "videos"
    return config, project, video_dir


def create_training_dataset(config_path, epochs=200, save_epochs=100, batch_size=8):
    config_path, project_path, _ = get_project_paths(config_path)
    deeplabcut.create_training_dataset(str(config_path))
    patch_pytorch_config(project_path, epochs, save_epochs, batch_size)

def patch_pytorch_config(project_path, epochs=200, save_epochs=100, batch_size=None, timeout=10):
    search_path = project_path / "dlc-models-pytorch" / "iteration-0"
    for _ in range(timeout * 2):
        config_files = list(search_path.glob("*/train/pytorch_config.yaml"))
        if config_files:
            break
        time.sleep(0.5)
    else:
        print("❌ pytorch_config.yaml not found.")
        return

    with open(config_files[0], "r") as f:
        cfg = yaml.safe_load(f)

    cfg["train_settings"]["epochs"] = epochs
    if batch_size:
        cfg["train_settings"]["batch_size"] = batch_size
    if "runner" in cfg and "snapshots" in cfg["runner"]:
        cfg["runner"]["snapshots"]["save_epochs"] = save_epochs

    with open(config_files[0], "w") as f:
        yaml.dump(cfg, f)

def train_network(config_path):
    deeplabcut.train_network(str(config_path), shuffle=1, displayiters=100, saveiters=1000, maxiters=20000)

def evaluate_network(config_path):
    deeplabcut.evaluate_network(str(config_path), plotting=True, show_errors=True)

def analyze_videos(config_path):
    _, project_path, video_dir = get_project_paths(config_path)
    videos = sorted([str(v) for v in video_dir.glob("*.mp4") if cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_WIDTH) > 0])
    deeplabcut.analyze_videos(str(config_path), videos=videos, shuffle=1, save_as_csv=True)
    for h5file in video_dir.glob("*shuffle1.h5"):
        if "_snapshot" not in h5file.name:
            new_name = h5file.with_name(h5file.stem + "_snapshot_070.h5")
            h5file.rename(new_name)

def create_labeled_videos(project_path):
    config_path, _, video_dir = get_project_paths(project_path)
    video_list = sorted([str(v) for v in video_dir.glob("*.mp4")])
    deeplabcut.create_labeled_video(str(config_path), videos=video_list, shuffle=1, draw_skeleton=True, pcutoff=0.3)

def create_conf_overlay(project_path):
    import deeplabcut
    import pandas as pd
    import cv2
    from pathlib import Path
    from tqdm import tqdm

    project = Path(project_path)
    video_dir = project / "videos"
    pcutoff = 0.0

    skeleton = [
        ("pupil_left_edge", "pupil_center"),
        ("pupil_right_edge", "pupil_center"),
        ("pupil_top_edge", "pupil_center"),
        ("pupil_bottom_edge", "pupil_center"),
    ]

    print("\n🎞️ Creating confidence overlay videos...")

    # Get all prediction files
    h5_files = sorted(video_dir.glob("*shuffle1_snapshot_070.h5"))

    for h5_path in h5_files:
        # Extract base_stem by stripping model suffix (everything after _stack_export)
        stem = h5_path.stem
        if "_stack_export" not in stem:
            print(f"⚠️ Skipping {h5_path.name}, unexpected filename format.")
            continue
        base_stem = stem.split("_stack_export")[0] + "_stack_export"

        video_path = video_dir / f"{base_stem}.mp4"
        if not video_path.exists():
            print(f"❌ Skipping, missing video: {video_path.name}")
            continue

        try:
            df = pd.read_hdf(h5_path)
        except Exception as e:
            print(f"❌ Error reading {h5_path.name}: {e}")
            continue

        if df.isnull().all().all():
            print(f"⚠️ Empty predictions in {h5_path.name}")
            continue

        scorer = df.columns.get_level_values(0)[0]
        bodyparts = list(set(df.columns.get_level_values(1)))

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = video_path.with_name(video_path.stem + "_labeled_conf.mp4")
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"🔧 Writing: {out_path.name}")

        for i in tqdm(range(frame_count), desc=video_path.stem, unit="frame"):
            success, frame = cap.read()
            if not success:
                break

            for bp in bodyparts:
                x = df[scorer][bp]["x"].iloc[i]
                y = df[scorer][bp]["y"].iloc[i]
                p = df[scorer][bp]["likelihood"].iloc[i]

                if p < pcutoff or pd.isna(x) or pd.isna(y):
                    continue

                color = (0, 0, 255) if p < 0.3 else (0, 255, 255) if p < 0.6 else (0, 255, 0)
                cv2.putText(frame, "*", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            for a, b in skeleton:
                if a in df[scorer] and b in df[scorer]:
                    p1 = df[scorer][a]["likelihood"].iloc[i]
                    p2 = df[scorer][b]["likelihood"].iloc[i]
                    if p1 >= pcutoff and p2 >= pcutoff:
                        x1 = int(df[scorer][a]["x"].iloc[i])
                        y1 = int(df[scorer][a]["y"].iloc[i])
                        x2 = int(df[scorer][b]["x"].iloc[i])
                        y2 = int(df[scorer][b]["y"].iloc[i])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            out.write(frame)

        cap.release()
        out.release()

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"✅ Saved: {out_path.name}")
        else:
            print(f"❌ Failed to save: {out_path.name}")

def reset_for_reanalysis(project_path):
    project = Path(project_path)
    video_dir = project / "videos"
    for pattern in ["*_labeled.mp4", "*_labeled_conf.mp4", "*.h5", "*.csv", "*full.pickle", "*meta.pickle"]:
        for f in video_dir.glob(pattern):
            if f.is_file():
                f.unlink()

    labeled_data = project / "labeled-data"
    if labeled_data.exists():
        for folder in labeled_data.glob("*"):
            if folder.is_dir():
                for f in folder.iterdir():
                    if f.is_file():
                        keep = (f.name.startswith("CollectedData_") and f.suffix in [".h5", ".csv"]) or f.suffix == ".png"
                        if not keep:
                            f.unlink()

def reset_for_retraining(project_path):
    project = Path(project_path)
    for folder in ["dlc-models", "dlc-models-pytorch", "training-datasets", "evaluation-results", "evaluation-results-pytorch"]:
        target = project / folder
        if target.exists():
            shutil.rmtree(target)
    reset_for_reanalysis(project)

def generate_training_plots(project_path):
    try:
        train_dir = next((Path(project_path) / "dlc-models-pytorch/iteration-0").glob("*train*"))
        csv_path = train_dir / "train" / "learning_stats.csv"
    except StopIteration:
        print("❌ Could not locate training directory.")
        return

    if not csv_path.exists():
        print("❌ Training CSV not found.")
        return

    df = pd.read_csv(csv_path)
    output_dir = Path(project_path) / "training_performance_summaries"
    output_dir.mkdir(exist_ok=True)
    output_pdf = output_dir / "training_summary.pdf"

    # Plot all numeric columns except 'step'
    all_cols = [col for col in df.columns if col != "step"]

    with PdfPages(output_pdf) as pdf:
        for col in all_cols:
            plt.figure()
            plt.plot(df['step'], df[col])
            plt.title(col)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True)
            pdf.savefig()
            plt.close()
            print(f"✅ Plotted: {col}")

    print(f"\n✅ Saved training performance PDF: {output_pdf}")

def get_config_path(project_path):
    path = Path(project_path) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found in {project_path}")
    return path

def train(project_path):
    config_path = get_config_path(project_path)
    create_training_dataset(config_path)
    train_network(config_path)
    evaluate_network(config_path)
    generate_training_plots(project_path)

def create_labeled(project_path):
    config_path = get_config_path(project_path)
    create_labeled_videos(config_path)

def extract_stimulus_frames(ttl_dir, video_dir):
    ttl_dir = Path(ttl_dir)
    video_dir = Path(video_dir)
    h5_files = sorted(ttl_dir.glob("TSeries_000*.h5"))

    for h5_path in h5_files:
        uid = h5_path.stem.split("_")[-1]
        base_output_dir = h5_path.parent / f"TSeries_{uid}"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        video_matches = list(video_dir.glob(f"*{uid}*.mp4"))
        if not video_matches:
            print(f"❌ No video found for {uid}")
            continue
        video_path = video_matches[0]

        plot_path = base_output_dir / f"ttl_combined_plot_{uid}.png"
        csv_path = base_output_dir / f"ttl_combined_summary_{uid}.csv"
        html_path = base_output_dir / f"stimulus_timing_summary_{uid}.html"
        pdf_path = base_output_dir / f"stimulus_timing_summary_{uid}.pdf"

        with h5py.File(h5_path, "r") as f:
            sample_rate = float(f["header/AcquisitionSampleRate"][0][0])
            sweep_keys = [key for key in f if key.startswith("sweep_")]
            if not sweep_keys:
                raise ValueError(f"No sweep group found in {h5_path.name}")
            sweep_group = sweep_keys[0]

            time = np.arange(f[f"{sweep_group}/digitalScans"].shape[1]) / sample_rate
            ttl = f[f"{sweep_group}/digitalScans"][0].astype(int)
            analog = f[f"{sweep_group}/analogScans"][2]

        stim_on_idx = np.where((ttl[:-1] == 1) & (ttl[1:] == 3))[0] + 1
        stim_off_idx = np.where((ttl[:-1] == 3) & (ttl[1:] == 1))[0] + 1
        stim_on_times = time[stim_on_idx]
        stim_off_times = time[stim_off_idx]

        cam_signal = (analog - np.min(analog)) / (np.max(analog) - np.min(analog))
        cam_rising = np.where(np.diff(cam_signal) > 0.5)[0] + 1
        cam_falling = np.where(np.diff(cam_signal) < -0.5)[0] + 1
        cam_start_time = time[cam_rising[0]] if cam_rising.size > 0 else np.nan
        cam_stop_time = time[cam_falling[-1]] if cam_falling.size > 0 else np.nan

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video for {uid}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if np.isnan(cam_start_time) or np.isnan(cam_stop_time):
            print(f"⚠️ Missing camera timing for {uid}")
            continue

        fps = total_frames / (cam_stop_time - cam_start_time)
        def time_to_frame(t): return int(round((t - cam_start_time) * fps))

        stim_data = [{
            "stimulus_index": i+1,
            "stimulus_on_time": on,
            "stimulus_off_time": off,
            "stimulus_on_frame": time_to_frame(on),
            "stimulus_off_frame": time_to_frame(off),
        } for i, (on, off) in enumerate(zip(stim_on_times, stim_off_times))]

        summary_df = pd.DataFrame(stim_data)
        if summary_df.empty:
            continue

        camera_row = pd.DataFrame([{
            "stimulus_index": "CAMERA",
            "stimulus_on_time": cam_start_time,
            "stimulus_off_time": cam_stop_time,
            "stimulus_on_frame": 0,
            "stimulus_off_frame": time_to_frame(cam_stop_time),
            "total_video_frames": total_frames,
            "fps": fps
        }])
        full_df = pd.concat([camera_row, summary_df], ignore_index=True)
        full_df.to_csv(csv_path, index=False)

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(time, ttl, label="TTL")
        plt.plot(time, cam_signal * 3, label="Analog Cam Signal", alpha=0.6)
        for on, off, onf, of in zip(stim_on_times, stim_off_times, summary_df["stimulus_on_frame"], summary_df["stimulus_off_frame"]):
            plt.plot(on, 1, "go")
            plt.text(on, 1.05, f"{onf}", color="green", fontsize=8)
            plt.plot(off, 3, "ro")
            plt.text(off, 3.05, f"{of}", color="red", fontsize=8)
        plt.axvline(cam_start_time, color="blue", linestyle="--", label="Cam Start")
        plt.axvline(cam_stop_time, color="purple", linestyle="--", label="Cam Stop")
        plt.xlabel("Time (s)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(plot_path); plt.close()

        # HTML + PDF
        with open(plot_path, "rb") as f: img64 = base64.b64encode(f.read()).decode()
        html = f"""<html><body><h1>TTL Report {uid}</h1>
        {full_df.to_html(index=False)}<br><img src="data:image/png;base64,{img64}"></body></html>"""
        with open(html_path, "w") as f: f.write(html)
        try:
            pisa.CreatePDF(html, open(pdf_path, "wb"))
        except:
            print("PDF conversion skipped (xhtml2pdf not available)")

def extract_stimulus_frames(ttl_dir, video_dir):
    ttl_dir = Path(ttl_dir)
    video_dir = Path(video_dir)
    h5_files = sorted(ttl_dir.glob("TSeries_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))

    print(f"\n📁 Starting stimulus extraction for {len(h5_files)} TSeries files...")

    for h5_path in h5_files:
        print(f"\n📄 Processing: {h5_path.name}")
        uid = h5_path.stem.split("_")[-1]
        base_output_dir = h5_path.parent / f"TSeries_{uid}"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        video_matches = list(video_dir.glob(f"*{uid}*.mp4"))
        if not video_matches:
            print(f"❌ No video found matching UID {uid}")
            continue
        else:
            video_path = video_matches[0]
            print(f"🎞️ Matched video: {video_path.name}")

        plot_path = base_output_dir / f"ttl_combined_plot_{uid}.png"
        csv_path = base_output_dir / f"ttl_combined_summary_{uid}.csv"
        filtered_csv_path = base_output_dir / f"ttl_filtered_window_{uid}.csv"
        html_path = base_output_dir / f"stimulus_timing_summary_{uid}.html"
        pdf_path = base_output_dir / f"stimulus_timing_summary_{uid}.pdf"

        with h5py.File(h5_path, "r") as f:
            sample_rate = float(f["header/AcquisitionSampleRate"][0][0])
            sweep_keys = [key for key in f if key.startswith("sweep_")]
            if not sweep_keys:
                print(f"❌ No sweep group in {h5_path.name}")
                continue
            sweep_group = sweep_keys[0]
            time = np.arange(f[f"{sweep_group}/digitalScans"].shape[1]) / sample_rate
            ttl = f[f"{sweep_group}/digitalScans"][0].astype(int)
            analog = f[f"{sweep_group}/analogScans"][2]

        stim_on_idx = np.where((ttl[:-1] == 1) & (ttl[1:] == 3))[0] + 1
        stim_off_idx = np.where((ttl[:-1] == 3) & (ttl[1:] == 1))[0] + 1
        stim_on_times = time[stim_on_idx]
        stim_off_times = time[stim_off_idx]

        if len(stim_on_times) < 4 or len(stim_off_times) < 4:
            print(f"⚠️ Not enough stimulus events in {h5_path.name}")
            continue

        cam_start_time = stim_on_times[0] - 0.01
        cam_stop_time = stim_off_times[-1] + 0.01

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video for {uid}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_duration_sec = cam_stop_time - cam_start_time
        fps = total_frames / video_duration_sec

        def time_to_frame(t): return int(round((t - cam_start_time) * fps))

        stim_data = [{
            "stimulus_index": i + 1,
            "stimulus_on_time": on,
            "stimulus_off_time": off,
            "stimulus_on_frame": time_to_frame(on),
            "stimulus_off_frame": time_to_frame(off),
        } for i, (on, off) in enumerate(zip(stim_on_times, stim_off_times))]

        summary_df = pd.DataFrame(stim_data)
        if summary_df.empty:
            print("⚠️ No stimulus events found, skipping.")
            continue

        camera_row = pd.DataFrame([{
            "stimulus_index": "CAMERA",
            "stimulus_on_time": cam_start_time,
            "stimulus_off_time": cam_stop_time,
            "stimulus_on_frame": 0,
            "stimulus_off_frame": time_to_frame(cam_stop_time),
            "total_video_frames": total_frames,
            "fps": fps
        }])
        full_df = pd.concat([camera_row, summary_df], ignore_index=True)
        full_df.to_csv(csv_path, index=False)
        print(f"✅ Saved full summary CSV: {csv_path.name}")

        start_frame = summary_df["stimulus_on_frame"].min()
        end_frame = summary_df["stimulus_off_frame"].max()
        filtered_df = full_df[
            (full_df["stimulus_on_frame"] >= start_frame) &
            (full_df["stimulus_off_frame"] <= end_frame) &
            (full_df["stimulus_index"] != "CAMERA")
        ]
        filtered_df.to_csv(filtered_csv_path, index=False)
        print(f"✅ Saved stimulus-only filtered CSV: {filtered_csv_path.name}")

        plt.figure(figsize=(14, 6))
        plt.plot(time, ttl, label="TTL")
        plt.plot(time, analog / np.max(analog) * 3, label="Analog Cam Signal", alpha=0.6)
        for on, off, onf, of in zip(stim_on_times, stim_off_times,
                                    summary_df["stimulus_on_frame"], summary_df["stimulus_off_frame"]):
            plt.plot(on, 1, "go")
            plt.text(on, 1.05, f"{onf}", color="green", fontsize=8)
            plt.plot(off, 3, "ro")
            plt.text(off, 3.05, f"{of}", color="red", fontsize=8)
        plt.axvline(cam_start_time, color="blue", linestyle="--", label="Cam Start (t=0)")
        plt.axvline(cam_stop_time, color="purple", linestyle="--", label="Cam Stop")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"🖼️ Saved TTL plot: {plot_path.name}")

        with open(plot_path, "rb") as f:
            img64 = base64.b64encode(f.read()).decode()
        wrapped_title = "<br>".join(textwrap.wrap(f"TTL Report {uid}", width=80))
        html = f"""<html><body><h1>{wrapped_title}</h1>
        {full_df.to_html(index=False)}<br><img src="data:image/png;base64,{img64}"></body></html>"""
        with open(html_path, "w") as f:
            f.write(html)
        print(f"📝 Saved HTML summary: {html_path.name}")
        try:
            pisa.CreatePDF(html, open(pdf_path, "wb"))
        except:
            pass
        print(f"📄 PDF export {'✅ success' if pdf_path.exists() else '⚠️ skipped'}: {pdf_path.name}")

def generate_eye_tracking_pdf(project_path):
    try:
        project_path = Path(project_path)
        video_dir = project_path / "videos"
        output_pdf = project_path / "eye_tracking_summary_plots.pdf"

        h5_files = sorted(video_dir.glob("*.h5"))
        if not h5_files:
            print(f"❌ No .h5 files found in: {video_dir}")
            return

        with PdfPages(output_pdf) as pdf:
            for h5_path in h5_files:
                try:
                    df = pd.read_hdf(h5_path)
                    if not isinstance(df.columns, pd.MultiIndex):
                        continue

                    model_name = df.columns.levels[0][0]
                    def get_col(part, coord):
                        return df[(model_name, part, coord)]

                    pupil_x = get_col("pupil_center", "x")
                    pupil_y = get_col("pupil_center", "y")
                    init_x, init_y = pupil_x.iloc[0], pupil_y.iloc[0]
                    displacement = np.sqrt((pupil_x - init_x)**2 + (pupil_y - init_y)**2)

                    left_x = get_col("pupil_left_edge", "x")
                    right_x = get_col("pupil_right_edge", "x")
                    top_y = get_col("pupil_top_edge", "y")
                    bottom_y = get_col("pupil_bottom_edge", "y")

                    diameter_h = np.abs(right_x - left_x)
                    diameter_v = np.abs(bottom_y - top_y)
                    diameter_avg = (diameter_h + diameter_v) / 2
                    frames = np.arange(len(df))

                    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    fig.suptitle(f"{h5_path.name}", fontsize=14)

                    axs[0].plot(frames, pupil_x, label='Pupil X', color='blue')
                    axs[0].plot(frames, pupil_y, label='Pupil Y', color='green')
                    axs[0].set_ylabel("Center Position (px)")
                    axs[0].legend(); axs[0].grid(True)

                    axs[1].plot(frames, displacement, label="Displacement", color='purple')
                    axs[1].set_ylabel("Movement (px)")
                    axs[1].legend(); axs[1].grid(True)

                    axs[2].plot(frames, diameter_h, label='Horizontal', color='orange')
                    axs[2].plot(frames, diameter_v, label='Vertical', color='red')
                    axs[2].set_ylabel("Diameter (px)")
                    axs[2].legend(); axs[2].grid(True)

                    axs[3].plot(frames, diameter_avg, label='Avg Diameter', color='black')
                    axs[3].set_xlabel("Frame #")
                    axs[3].set_ylabel("Avg Diameter (px)")
                    axs[3].legend(); axs[3].grid(True)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)

                except Exception as e:
                    print(f"⚠️ Skipped {h5_path.name}: {e}")

        print(f"✅ Saved: {output_pdf}")

    except Exception as err:
        print(f"❌ PDF generation failed: {err}")

def generate_aligned_analysis(project_path):
    project_path = Path(project_path)
    video_dir = project_path / "videos"
    stim_dir = project_path / "stimulus"
    output_dir = project_path / "aligned_analysis"
    export_dir = output_dir / "exports"
    output_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)

    def extract_uid(path):
        match = re.search(r"_(\d{4})_stack_export", path.stem)
        return match.group(1) if match else None

    # Map UID to files
    video_map = {extract_uid(p): p for p in video_dir.glob("*.h5") if extract_uid(p)}
    stim_map = {f.name.split("_")[-1]: f for f in stim_dir.glob("TSeries_*") if f.is_dir()}
    common_uids = sorted(set(video_map) & set(stim_map))

    global_pdf_path = output_dir / "stimulus_overlay_summary.pdf"
    global_pdf = PdfPages(global_pdf_path)

    for uid in common_uids:
        video_h5 = video_map[uid]
        stim_folder = stim_map[uid]

        try:
            df = pd.read_hdf(video_h5)
            model_prefix = df.columns.levels[0][0]
        except Exception as e:
            print(f"❌ Failed to read {video_h5.name}: {e}")
            continue

        base_name = video_h5.stem
        def get(bp, coord): return df[(model_prefix, bp, coord)]
        pupil_x = get("pupil_center", "x")
        pupil_y = get("pupil_center", "y")
        displacement = np.sqrt((pupil_x - pupil_x.iloc[0])**2 + (pupil_y - pupil_y.iloc[0])**2)
        diam_h = np.abs(get("pupil_right_edge", "x") - get("pupil_left_edge", "x"))
        diam_v = np.abs(get("pupil_bottom_edge", "y") - get("pupil_top_edge", "y"))
        diam_avg = (diam_h + diam_v) / 2
        frames = np.arange(len(df))

        stim_csvs = list(stim_folder.glob("ttl_combined_summary_*.csv"))
        if not stim_csvs:
            print(f"⚠️ No TTL CSV in {stim_folder.name}")
            continue

        try:
            stim_df = pd.read_csv(stim_csvs[0])
            stim_df = stim_df[stim_df["stimulus_index"] != "CAMERA"]
        except Exception as e:
            print(f"❌ Failed to read CSV in {stim_folder.name}: {e}")
            continue

        # Build full DataFrame across all frames
        aligned_df = pd.DataFrame({
            "frame_number": frames,
            "pupil_x": pupil_x,
            "pupil_y": pupil_y,
            "movement": displacement,
            "diameter_h": diam_h,
            "diameter_v": diam_v,
            "diameter_avg": diam_avg,
            "stimulus_index": np.nan,
            "video_name": base_name
        })

        # Annotate stimulus windows in the full trace
        for _, row in stim_df.iterrows():
            stim_num = int(row["stimulus_index"])
            start = int(row["stimulus_on_frame"])
            stop = int(row["stimulus_off_frame"])
            aligned_df.loc[(aligned_df["frame_number"] >= start) & (aligned_df["frame_number"] < stop), "stimulus_index"] = stim_num

        # ✅ Save the aligned trace for this video
        aligned_csv_path = export_dir / f"{base_name}_aligned_raw.csv"
        aligned_df.to_csv(aligned_csv_path, index=False)

        # Plot overlay
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        wrapped_title = "\n".join(textwrap.wrap(f"{base_name} — Stimulus Overlay", 80))
        fig.suptitle(wrapped_title, fontsize=12)

        axs[0].plot(frames, pupil_x, label="Pupil X", color="blue")
        axs[0].plot(frames, pupil_y, label="Pupil Y", color="green")
        axs[0].set_ylabel("Center Pos (px)")
        axs[0].legend(); axs[0].grid(True)

        axs[1].plot(frames, displacement, label="Displacement", color="purple")
        axs[1].set_ylabel("Movement (px)")
        axs[1].legend(); axs[1].grid(True)

        axs[2].plot(frames, diam_h, label="Diameter H", color="orange")
        axs[2].plot(frames, diam_v, label="Diameter V", color="red")
        axs[2].set_ylabel("Diameter (px)")
        axs[2].legend(); axs[2].grid(True)

        axs[3].plot(frames, diam_avg, label="Avg Diameter", color="black")
        axs[3].set_xlabel("Frame")
        axs[3].set_ylabel("Avg Diameter (px)")
        axs[3].legend(); axs[3].grid(True)

        for _, row in stim_df.iterrows():
            on = int(row["stimulus_on_frame"])
            off = int(row["stimulus_off_frame"])
            for ax in axs:
                ax.axvline(on, color="green", linestyle="--", alpha=0.7)
                ax.axvline(off, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf_path = output_dir / f"{base_name}_stimulus_overlay.pdf"
        with PdfPages(pdf_path) as pdf_indiv:
            pdf_indiv.savefig(fig)
        global_pdf.savefig(fig)
        plt.close(fig)

        print(f"✅ Saved aligned CSV and overlay PDF for UID {uid}")

    global_pdf.close()
    print(f"\n✅ Global overlay PDF saved: {global_pdf_path}")

def detect_saccades_and_dilation(project_path, saccade_min_amp=5.0, saccade_max_amp=40.0, saccade_min_width=2, dilation_min_width=5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.signal import find_peaks
    from pathlib import Path
    import re
    import textwrap

    print("\n🔍 Detecting saccades and pupil dilation events...")
    project_path = Path(project_path)
    export_dir = project_path / "aligned_analysis" / "exports"
    output_dir = project_path / "aligned_analysis" / "saccades"
    output_dir.mkdir(exist_ok=True)

    raw_csvs = sorted(export_dir.glob("*_aligned_raw.csv"), key=lambda p: int(re.search(r"_(\d{4})", p.stem).group(1)))
    global_pdf_path = output_dir / "saccade_dilation_overlay_summary.pdf"
    global_pdf = PdfPages(global_pdf_path)

    for csv_path in raw_csvs:
        print(f"\n📄 Processing: {csv_path.stem}")
        df = pd.read_csv(csv_path)
        base_name = csv_path.stem.replace("_aligned_raw", "")

        if "movement" not in df.columns or "stimulus_index" not in df.columns:
            print(f"⚠️ Skipping {base_name}: required columns missing.")
            continue

        # === Detect saccades using displacement peaks ===
        displacement = df["movement"].values
        saccade_peaks, _ = find_peaks(
            displacement,
            height=(saccade_min_amp, saccade_max_amp),
            width=saccade_min_width
        )
        df["saccade"] = False
        df.loc[saccade_peaks, "saccade"] = True

        # === Detect pupil dilation during stimulus 4 using diameter_avg ===
        df["dilation"] = False
        for stim_index in df["stimulus_index"].dropna().unique():
            if int(stim_index) != 4:
                continue
            stim_df = df[df["stimulus_index"] == stim_index]
            diam_avg = stim_df["diameter_avg"].values
            dilation_peaks, _ = find_peaks(diam_avg, width=dilation_min_width)
            df.loc[stim_df.iloc[dilation_peaks].index, "dilation"] = True

        # Save results
        output_csv = output_dir / f"{base_name}_saccades_dilation.csv"
        df.to_csv(output_csv, index=False)

        # === Plotting ===
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        wrapped_title = "\n".join(textwrap.wrap(f"{base_name} — Saccade and Dilation Detection", 80))
        fig.suptitle(wrapped_title, fontsize=12)

        axs[0].plot(df["frame_number"], df["pupil_x"], label="Pupil X", color="blue")
        axs[0].plot(df["frame_number"], df["pupil_y"], label="Pupil Y", color="green")
        axs[0].set_ylabel("Center Pos (px)")
        axs[0].legend()

        axs[1].plot(df["frame_number"], df["movement"], label="Displacement", color="purple")
        axs[1].scatter(df[df["saccade"]]["frame_number"], df[df["saccade"]]["movement"], label="Saccade", color="red", marker="v", s=12, alpha=0.7)
        axs[1].set_ylabel("Displacement (px)")
        axs[1].legend()

        axs[2].plot(df["frame_number"], df["diameter_h"], label="Diameter H", color="orange")
        axs[2].plot(df["frame_number"], df["diameter_v"], label="Diameter V", color="red")
        axs[2].set_ylabel("Diameter (px)")
        axs[2].legend()

        axs[3].plot(df["frame_number"], df["diameter_avg"], label="Avg Diameter", color="black")
        axs[3].scatter(df[df["dilation"]]["frame_number"], df[df["dilation"]]["diameter_avg"], label="Dilation", color="blue", marker="o", s=12, alpha=0.7)
        axs[3].set_xlabel("Frame")
        axs[3].set_ylabel("Avg Diameter (px)")
        axs[3].legend()

        # Annotate stimulus transitions
        stim_df = df.dropna(subset=["stimulus_index"])
        stim_changes = []
        prev_stim = None
        for _, row in stim_df.iterrows():
            stim = row["stimulus_index"]
            if stim != prev_stim:
                stim_changes.append(int(row["frame_number"]))
                prev_stim = stim

        for x in stim_changes:
            for ax in axs:
                ax.axvline(x, color="gray", linestyle=":", alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf_path = output_dir / f"{base_name}_saccade_dilation_overlay.pdf"
        with PdfPages(pdf_path) as pdf_indiv:
            pdf_indiv.savefig(fig)
        global_pdf.savefig(fig)
        plt.close(fig)

        print(f"✅ Saved saccade+dilation CSV and plot for: {base_name}")

    global_pdf.close()
    print(f"\n✅ Global overlay PDF saved: {global_pdf_path}")

def interactive_trace_viewer(project_path):
    import pandas as pd
    import plotly.graph_objects as go
    from pathlib import Path
    import re

    project_path = Path(project_path)
    export_dir = project_path / "aligned_analysis" / "exports"
    raw_csvs = sorted(export_dir.glob("*_aligned_raw.csv"), key=lambda p: int(re.search(r"_(\d{4})", p.stem).group(1)))

    for csv_path in raw_csvs:
        df = pd.read_csv(csv_path)
        base_name = csv_path.stem.replace("_aligned_raw", "")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["pupil_x"], mode='lines', name="Pupil X"))
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["pupil_y"], mode='lines', name="Pupil Y"))
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["movement"], mode='lines', name="Displacement"))
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["diameter_h"], mode='lines', name="Diameter H"))
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["diameter_v"], mode='lines', name="Diameter V"))
        fig.add_trace(go.Scatter(x=df["frame_number"], y=df["diameter_avg"], mode='lines', name="Avg Diameter"))

        # Optional: overlay stimulus change vertical lines
        stim_df = df.dropna(subset=["stimulus_index"])
        stim_changes = []
        prev_stim = None
        for _, row in stim_df.iterrows():
            stim = row["stimulus_index"]
            if stim != prev_stim:
                stim_changes.append(int(row["frame_number"]))
                prev_stim = stim

        for x in stim_changes:
            fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="gray")

        fig.update_layout(
            title=f"Interactive Trace Viewer: {base_name}",
            xaxis_title="Frame Number",
            yaxis_title="Value",
            hovermode="x unified"
        )

        fig.show()