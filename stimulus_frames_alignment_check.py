import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re

# Set your DLC project path
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")
video_dir = project_path / "videos"
stim_dir = project_path / "stimulus"
output_pdf = project_path / "pupil_with_stimulus_frames.pdf"
output_pdf.parent.mkdir(parents=True, exist_ok=True)

# Match video and stimulus folders by 4-digit index
def extract_index(stem):
    match = re.search(r"_(\d{4})", stem)
    return int(match.group(1)) if match else float('inf')

video_h5_files = sorted(video_dir.glob("*.h5"), key=lambda x: extract_index(x.stem))
tseries_folders = sorted([f for f in stim_dir.glob("TSeries_*") if f.is_dir()], key=lambda x: extract_index(x.name))

with PdfPages(output_pdf) as pdf:
    for video_h5, stim_folder in zip(video_h5_files, tseries_folders):
        try:
            df = pd.read_hdf(video_h5)
            model_prefix = df.columns.levels[0][0]
            def get(part, coord): return df[(model_prefix, part, coord)]
        except Exception as e:
            print(f"❌ Failed to read {video_h5.name}: {e}")
            continue

        # Match by UID index
        uid = extract_index(video_h5.stem)
        ttl_csvs = list(stim_folder.glob("ttl_combined_summary_*.csv"))
        if not ttl_csvs:
            continue

        try:
            stim_df = pd.read_csv(ttl_csvs[0])
            stim_df = stim_df[stim_df["stimulus_index"] != "CAMERA"]
        except Exception as e:
            print(f"❌ Failed to load TTL CSV for {stim_folder.name}: {e}")
            continue

        # Extract pupil metrics
        pupil_x = get("pupil_center", "x")
        pupil_y = get("pupil_center", "y")
        displacement = np.sqrt((pupil_x - pupil_x.iloc[0])**2 + (pupil_y - pupil_y.iloc[0])**2)
        diam_h = np.abs(get("pupil_right_edge", "x") - get("pupil_left_edge", "x"))
        diam_v = np.abs(get("pupil_bottom_edge", "y") - get("pupil_top_edge", "y"))
        diam_avg = (diam_h + diam_v) / 2
        frames = np.arange(len(df))

        # Plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{video_h5.name}", fontsize=14)

        axs[0].plot(frames, displacement, label="Displacement", color="purple")
        axs[0].set_ylabel("Displacement (px)")
        axs[0].legend(); axs[0].grid(True)

        axs[1].plot(frames, diam_h, label="Diameter H", color="orange")
        axs[1].plot(frames, diam_v, label="Diameter V", color="red")
        axs[1].set_ylabel("Diameter (px)")
        axs[1].legend(); axs[1].grid(True)

        axs[2].plot(frames, diam_avg, label="Avg Diameter", color="black")
        axs[2].set_xlabel("Frame #")
        axs[2].set_ylabel("Avg Diameter (px)")
        axs[2].legend(); axs[2].grid(True)

        # Mark stimulus frames
        for _, row in stim_df.iterrows():
            on_f = int(row["stimulus_on_frame"])
            off_f = int(row["stimulus_off_frame"])
            for ax in axs:
                ax.axvline(on_f, color="green", linestyle="--", alpha=0.7)
                ax.axvline(off_f, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

print(f"✅ Saved to: {output_pdf}")
