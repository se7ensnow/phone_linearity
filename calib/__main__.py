
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import load_recording, process_file, process_session
from .model import AccelerometerCalibrator
from .evaluation import acceleration_errors, tilt_angle_error_deg


G = 9.786

POSES = {
    "screen_up": [0, 0, G],
    "screen_down": [0, 0, -G],
    "left_edge_down": [G, 0, 0],
    "right_edge_down": [-G, 0, 0],
    "bottom_edge_down": [0, G, 0],
    "top_edge_down": [0, -G, 0],
}


def load_session(data_dir, session):
    """Load six poses and return true and measured vectors."""
    suffix = "" if session == 1 else "_2"

    files = {
        pose: data_dir / f"{pose}{suffix}.csv"
        for pose in POSES
    }

    true_vectors = np.array(list(POSES.values()), dtype=float)
    measured_vectors = process_session(files, list(POSES))

    return true_vectors, measured_vectors


def run_fit(args):
    A, M = load_session(args.data_dir, session=1)

    model = AccelerometerCalibrator()
    model.fit(A, M)

    args.params.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.params)

    print("S:")
    print(model.S)

    print("\nb:")
    print(model.b)

    print("\nRank:", model.rank)
    print("Condition number:", model.condition_number)
    print(f"\nParameters saved to {args.params}")

    residuals = M - model.predict(A)
    print(
        "Mean residual norm:",
        np.linalg.norm(residuals, axis=1).mean(),
        "m/s²"
    )


def run_evaluate(args):
    model = AccelerometerCalibrator.load(args.params)
    A, M = load_session(args.data_dir, session=2)

    corrected = model.apply(M)

    results = pd.DataFrame({
        "error_before": acceleration_errors(M, A),
        "error_after": acceleration_errors(corrected, A),
        "angle_before": tilt_angle_error_deg(M, A),
        "angle_after": tilt_angle_error_deg(corrected, A),
    }, index=list(POSES))

    print("\nHeld-out evaluation (Session 2):")
    print(results.round(4).to_string())

    print("\nMean errors:")
    print(results.mean().round(4).to_string())

    if args.plot:
        sns.set_theme(style="whitegrid")

        ax = results[
            ["error_before", "error_after"]
        ].plot.bar(figsize=(10, 5))

        ax.set_title("Calibration on held-out poses")
        ax.set_ylabel("Acceleration error (m/s²)")
        ax.set_xlabel("Phone orientation")
        ax.legend(["Before", "After"])

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()


def run_apply(args):
    model = AccelerometerCalibrator.load(args.params)

    df = load_recording(args.recording)

    columns = [
        "Acceleration x (m/s^2)",
        "Acceleration y (m/s^2)",
        "Acceleration z (m/s^2)",
    ]

    raw = df[columns].to_numpy(dtype=float)
    corrected = model.apply(raw)

    output = pd.DataFrame(
        corrected,
        columns=["corrected_x", "corrected_y", "corrected_z"]
    )
    output.insert(0, "Time (s)", df["Time (s)"].to_numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Corrected recording saved to {args.output}")

    # Also show mean corrected acceleration at rest.
    result = process_file(df)
    mean_corrected = model.apply(result["mean"])

    print("\nMean corrected acceleration at rest:")
    print(mean_corrected)


def main():
    parser = argparse.ArgumentParser(
        description="Phone accelerometer calibration"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw")
    )

    parser.add_argument(
        "--params",
        type=Path,
        default=Path("results/params.json")
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    subparsers.add_parser("fit")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument(
        "--plot",
        action="store_true"
    )

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("recording", type=Path)
    apply_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/corrected.csv")
    )

    args = parser.parse_args()

    if args.command == "fit":
        run_fit(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "apply":
        run_apply(args)


if __name__ == "__main__":
    main()
