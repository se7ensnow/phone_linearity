import json
import numpy as np


class AccelerometerCalibrator:
    def __init__(self):
        self.S = None
        self.b = None
        self.rank = None
        self.condition_number = None

    def fit(self, true_vectors, measured_vectors):
        A = np.asarray(true_vectors, dtype=float)
        M = np.asarray(measured_vectors, dtype=float)

        if A.ndim != 2 or A.shape[1] != 3 or M.shape != A.shape:
            raise ValueError("Expected matching (N, 3) arrays")

        X = np.column_stack([A, np.ones(len(A))])

        self.rank = np.linalg.matrix_rank(X)
        self.condition_number = np.linalg.cond(X)

        if self.rank < 4:
            raise ValueError("Insufficient independent poses")

        theta, *_ = np.linalg.lstsq(X, M, rcond=None)

        self.S = theta[:3].T
        self.b = theta[3]

        if np.linalg.matrix_rank(self.S) < 3:
            raise ValueError("Estimated S is singular")

        return self

    def apply(self, measured):
        if self.S is None:
            raise ValueError("Model is not fitted")

        M = np.asarray(measured, dtype=float)
        return np.linalg.solve(self.S, (M - self.b).T).T

    def predict(self, true_vectors):
        """Predict raw sensor readings using the affine model."""
        A = np.asarray(true_vectors, dtype=float)
        return A @ self.S.T + self.b

    def save(self, path):
        if self.S is None:
            raise ValueError("Model is not fitted")

        with open(path, "w") as f:
            json.dump({
                "S": self.S.tolist(),
                "b": self.b.tolist()
            }, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            params = json.load(f)

        model = cls()
        model.S = np.asarray(params["S"], dtype=float)
        model.b = np.asarray(params["b"], dtype=float)
        return model
