import numpy as np
import pandas as pd

TRIM = 1.0
WINDOW = 1.0
THRESHOLD = 0.05

COLS = {
    "Time (s)": "t",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Absolute acceleration (m/s^2)": "abs",
}


def load_recording(path):
    df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    df.columns = df.columns.str.strip()
    return df


def process_file(df, trim=TRIM, threshold=THRESHOLD, window=WINDOW):
    """Extract stationary measurements from one recording."""
    data = df.rename(columns=COLS).copy()

    start_time = data["t"].min()
    data["w"] = ((data["t"] - start_time) // window).astype(int)

    # Standard deviation in each time window
    stats = data.groupby("w")[["x", "y", "z"]].std()
    stats["max_std"] = stats.max(axis=1)
    stats["still"] = stats["max_std"] < threshold

    # Exclude first and last seconds
    inside = (
        (data["t"] >= start_time + trim)
        & (data["t"] <= data["t"].max() - trim)
    )

    keep = inside & data["w"].map(stats["still"]).fillna(False)
    used = data[keep]

    if used.empty:
        raise ValueError("No stationary measurements found")

    w_inside = data.loc[inside, "w"].unique()
    bad = [
        int(w) for w in w_inside
        if not stats.loc[w, "still"]
    ]

    return {
        "mean": used[["x", "y", "z"]].mean().to_numpy(),
        "std": used[["x", "y", "z"]].std().to_numpy(),
        "rows_file": len(data),
        "rows_used": len(used),
        "max_wobble": stats.loc[w_inside, "max_std"].max(),
        "bad_windows": bad,
        "data": data,
        "stats": stats,
    }


def process_session(files, pose_names):
    """Process pose recordings in a fixed order."""
    means = []

    for name in pose_names:
        df = load_recording(files[name])
        result = process_file(df)
        means.append(result["mean"])

    return np.array(means)
