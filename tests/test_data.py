import numpy as np
import pandas as pd
import pytest

from calib.data import load_recording, process_file, process_session


def make_recording(n=600, moving=False):
    """Create a synthetic 6-second recording at 100 Hz."""
    t = np.arange(n) / 100

    acceleration = np.tile([0.1, -9.75, 0.2], (n, 1))

    # Small sensor noise
    rng = np.random.default_rng(42)
    acceleration += rng.normal(0, 0.01, (n, 3))

    if moving:
        # Strong oscillations in the third second
        acceleration[200:300, 0] += (
            np.arange(100) % 2
        ) * 2

    return pd.DataFrame({
        "Time (s)": t,
        "Acceleration x (m/s^2)": acceleration[:, 0],
        "Acceleration y (m/s^2)": acceleration[:, 1],
        "Acceleration z (m/s^2)": acceleration[:, 2],
    })


def test_load_recording(tmp_path):
    df = make_recording()
    path = tmp_path / "recording.csv"

    df.to_csv(path, sep=";", decimal=",", index=False)

    loaded = load_recording(path)

    assert len(loaded) == len(df)
    np.testing.assert_allclose(
        loaded["Acceleration x (m/s^2)"],
        df["Acceleration x (m/s^2)"]
    )


def test_stationary_recording():
    df = make_recording()
    result = process_file(df)

    assert result["rows_used"] > 0
    assert result["bad_windows"] == []

    np.testing.assert_allclose(
        result["mean"],
        [0.1, -9.75, 0.2],
        atol=0.01
    )


def test_moving_window_rejected():
    df = make_recording(moving=True)
    result = process_file(df)

    assert 2 in result["bad_windows"]
    assert result["rows_used"] < 400


def test_no_stationary_measurements():
    df = make_recording()

    # Make every window highly unstable
    df["Acceleration x (m/s^2)"] += (
        np.arange(len(df)) % 2
    ) * 2

    with pytest.raises(
        ValueError,
        match="No stationary measurements"
    ):
        process_file(df)


def test_process_session(tmp_path):
    df = make_recording()

    files = {}
    for pose in ["pose_a", "pose_b"]:
        path = tmp_path / f"{pose}.csv"
        df.to_csv(path, sep=";", decimal=",", index=False)
        files[pose] = path

    means = process_session(
        files, ["pose_a", "pose_b"]
    )

    assert means.shape == (2, 3)

    np.testing.assert_allclose(
        means[0], means[1]
    )
