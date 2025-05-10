import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def view_interactive_traces(csv_path):
    df = pd.read_csv(csv_path)
    frames = df["frame_number"]

    pupil_x = df["pupil_x"]
    pupil_y = df["pupil_y"]
    movement = df["movement"]
    diam_h = df["diameter_h"]
    diam_v = df["diameter_v"]
    diam_avg = df["diameter_avg"]

    # Enable interactive plotting
    plt.ion()

    # --- Figure 1: Pupil X & Y
    plt.figure("Pupil Position")
    plt.plot(frames, pupil_x, label="pupil_x", color="blue")
    plt.plot(frames, pupil_y, label="pupil_y", color="green")
    plt.title("Pupil Center Position (Interactive)")
    plt.xlabel("Frame")
    plt.ylabel("Position (px)")
    plt.legend()
    plt.grid(True)

    # --- Figure 2: Displacement
    plt.figure("Displacement")
    plt.plot(frames, movement, label="displacement", color="purple")
    plt.title("Displacement (Interactive)")
    plt.xlabel("Frame")
    plt.ylabel("Displacement (px)")
    plt.grid(True)

    # --- Figure 3: Diameters
    plt.figure("Diameters")
    plt.plot(frames, diam_h, label="diameter_h", color="orange")
    plt.plot(frames, diam_v, label="diameter_v", color="red")
    plt.plot(frames, diam_avg, label="diameter_avg", color="black")
    plt.title("Pupil Diameter (Interactive)")
    plt.xlabel("Frame")
    plt.ylabel("Diameter (px)")
    plt.legend()
    plt.grid(True)

    # --- Figure 4: Combined Overview
    plt.figure("Combined Trace Overview")
    plt.plot(frames, movement, label="movement", color="purple", alpha=0.6)
    plt.plot(frames, diam_avg, label="diameter_avg", color="black", alpha=0.6)
    plt.plot(frames, pupil_x, label="pupil_x", color="blue", alpha=0.4)
    plt.plot(frames, pupil_y, label="pupil_y", color="green", alpha=0.4)
    plt.title("All Traces Combined (Interactive)")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    print("👁️ Use zoom/pan tools to explore plots. Close all windows to exit.")
    plt.show(block=True)

# Example usage:
view_interactive_traces("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/aligned_analysis/exports/eye_recording_20250424_120209_0006_stack_exportDLC_Resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1_snapshot_070_aligned_raw.csv")
