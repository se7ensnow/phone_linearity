import numpy as np
import pytest

from calib.model import AccelerometerCalibrator


@pytest.fixture
def synthetic_data():
    g = 9.786

    A = np.array([
        [0, 0, g],
        [0, 0, -g],
        [g, 0, 0],
        [-g, 0, 0],
        [0, g, 0],
        [0, -g, 0],
    ])

    S_true = np.array([
        [1.02, 0.01, -0.005],
        [0.003, 0.98, 0.008],
        [-0.002, 0.004, 1.01],
    ])

    b_true = np.array([0.1, -0.05, 0.2])

    M = A @ S_true.T + b_true

    return A, M, S_true, b_true


def test_fit_recovers_parameters(synthetic_data):
    A, M, S_true, b_true = synthetic_data

    model = AccelerometerCalibrator().fit(A, M)

    np.testing.assert_allclose(
        model.S, S_true, atol=1e-10
    )

    np.testing.assert_allclose(
        model.b, b_true, atol=1e-10
    )

    assert model.rank == 4
    assert np.isfinite(model.condition_number)


def test_apply_corrects_measurements(synthetic_data):
    A, M, _, _ = synthetic_data

    model = AccelerometerCalibrator().fit(A, M)
    corrected = model.apply(M)

    np.testing.assert_allclose(
        corrected, A, atol=1e-10
    )


def test_save_and_load(synthetic_data, tmp_path):
    A, M, _, _ = synthetic_data

    model = AccelerometerCalibrator().fit(A, M)

    path = tmp_path / "params.json"
    model.save(path)

    assert path.exists()

    loaded = AccelerometerCalibrator.load(path)

    np.testing.assert_allclose(loaded.S, model.S)
    np.testing.assert_allclose(loaded.b, model.b)
    np.testing.assert_allclose(
        loaded.apply(M), A, atol=1e-10
    )


def test_insufficient_rank():
    # Only two distinct directions are measured
    A = np.array([
        [0, 0, 9.786],
        [0, 0, -9.786],
        [0, 0, 9.786],
        [0, 0, -9.786],
    ])

    M = A.copy()

    model = AccelerometerCalibrator()

    with pytest.raises(
        ValueError,
        match="Insufficient independent poses"
    ):
        model.fit(A, M)
