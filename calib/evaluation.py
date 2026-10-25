import numpy as np


def acceleration_errors(estimated, true):
    """Acceleration error for each sample, m/s²."""
    return np.linalg.norm(
        np.asarray(estimated) - np.asarray(true),
        axis=-1
    )


def tilt_angle_error_deg(estimated, true):
    """Angle between vectors, in degrees."""
    estimated = np.atleast_2d(estimated)
    true = np.atleast_2d(true)

    cosang = np.sum(estimated * true, axis=1) / (
        np.linalg.norm(estimated, axis=1)
        * np.linalg.norm(true, axis=1)
    )

    return np.degrees(
        np.arccos(np.clip(cosang, -1, 1))
    )


def pose_errors(estimated, true):
    """Mean acceleration and angular errors."""
    norm = acceleration_errors(estimated, true).mean()
    angle = tilt_angle_error_deg(estimated, true).mean()
    return norm, angle


def ramp_errors(estimated, true, theta_true):
    """Mean acceleration and board-angle errors."""
    tilt = np.degrees(
        np.arctan2(estimated[:, 1], estimated[:, 2])
    )

    norm = acceleration_errors(estimated, true).mean()
    angle = np.abs(tilt - theta_true).mean()

    return norm, angle


def rank_report(A, indices):
    """Rank of the design matrix for selected poses."""
    X = np.column_stack([A, np.ones(len(A))])
    subset = X[indices]

    return {
        "rank": np.linalg.matrix_rank(subset),
        "condition_number": np.linalg.cond(subset),
    }
