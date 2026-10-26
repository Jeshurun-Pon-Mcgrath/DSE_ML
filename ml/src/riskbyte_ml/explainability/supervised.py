"""Optional genuine Tree SHAP adapter; unavailable dependencies never yield fake values."""

from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TreeShap:
    def __init__(self, estimator, names):
        if not isinstance(estimator, DecisionTreeClassifier):
            raise ValueError("Frozen Decision Tree required")
        if len(names) != estimator.n_features_in_ or len(set(names)) != len(names):
            raise ValueError("SHAP feature alignment failed")
        classes = list(estimator.classes_)
        if classes != [0, 1]:
            raise ValueError("Explicit binary class contract required")
        self.estimator, self.names, self.positive_index = estimator, names, classes.index(1)
        self.explainer, self.reason, self.initialization_seconds = None, None, None
        try:
            import shap
        except ImportError:
            self.reason = "shap_dependency_unavailable"
            self.library_version = None
            return
        start = perf_counter()
        self.explainer = shap.TreeExplainer(
            estimator,
            feature_names=names,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        self.initialization_seconds = perf_counter() - start
        self.library_version = shap.__version__
        self.observed_layout = None

    @property
    def status(self):
        return "AVAILABLE" if self.explainer is not None else "UNAVAILABLE_WITH_REASON"

    def values(self, matrix):
        if matrix.shape[1] != len(self.names):
            raise ValueError("SHAP matrix is misaligned")
        if self.explainer is None:
            return None
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        output = self.explainer.shap_values(dense, check_additivity=True)
        values, base = normalize_output(
            output, self.explainer.expected_value, dense.shape, self.positive_index
        )
        self.observed_layout = "list" if isinstance(output, list) else f"{np.asarray(output).ndim}D"
        raw = self.estimator.predict_proba(dense)[:, self.positive_index]
        residual = base + values.sum(axis=1) - raw
        if not np.allclose(residual, 0, atol=1e-7, rtol=0):
            raise ValueError("SHAP must decompose raw tree probability, not calibrated P")
        return values, base, residual


def normalize_output(output, expected, shape, positive_index=1):
    """Normalize list, 2D and sample/feature/class 3D layouts; verify raw output separately."""
    if isinstance(output, list):
        if len(output) <= positive_index:
            raise ValueError("SHAP class output is missing")
        values = np.asarray(output[positive_index])
    else:
        values = np.asarray(output)
        if values.ndim == 3:
            if values.shape[2] <= positive_index:
                raise ValueError("SHAP class dimension is missing")
            values = values[:, :, positive_index]
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if len(expected) not in {1, 2}:
        raise ValueError("Ambiguous SHAP expected-value semantics")
    base = float(expected[positive_index] if len(expected) == 2 else expected[0])
    if values.shape != shape or not np.isfinite(values).all() or not np.isfinite(base):
        raise ValueError("Invalid Tree SHAP dimensions or numbers")
    return values, base


def measure_latency(callable_, repeats=3):
    if repeats < 1:
        raise ValueError("Positive repeat count required")
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        callable_()
        durations.append(perf_counter() - start)
    return {
        "repeats": repeats,
        "median_seconds": float(np.median(durations)),
        "minimum_seconds": min(durations),
        "maximum_seconds": max(durations),
    }
