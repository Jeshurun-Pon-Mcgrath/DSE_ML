"""Restrained reproducible searches and explicitly budgeted representative KNN."""

import importlib.util
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.models.supervised.metrics import metrics
from riskbyte_ml.models.supervised.schemas import KNNBenchmark
from riskbyte_ml.preprocessing.pipeline import matrix_summary

ALGORITHMS = ("Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost")


def representative_rows(labels, count, seed):
    if count >= len(labels):
        return np.arange(len(labels))
    selected, _ = train_test_split(
        np.arange(len(labels)), train_size=count, stratify=labels, random_state=seed
    )
    return np.sort(selected)


def make_estimator(algorithm, parameters, config):
    if algorithm == "Logistic Regression":
        return LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=config.seed, **parameters
        )
    if algorithm == "Decision Tree":
        return DecisionTreeClassifier(random_state=config.seed, **parameters)
    if algorithm == "KNN":
        return KNeighborsClassifier(algorithm="brute", n_jobs=config.n_jobs, **parameters)
    if algorithm == "Random Forest":
        # Serial forest aggregation keeps reloaded probabilities/threshold ties exact.
        return RandomForestClassifier(random_state=config.seed, n_jobs=1, **parameters)
    if algorithm == "XGBoost":
        if importlib.util.find_spec("xgboost") is None:
            raise DatasetError(
                "model_dependency_unavailable",
                "XGBoost is not installed; install the optional boosting dependency",
            )
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=config.seed,
            n_jobs=config.n_jobs,
            **parameters,
        )
    raise ValueError("Unsupported supervised algorithm")


def search_parameters(algorithm, labels, config):
    ratio = float(np.sum(labels == 0) / np.sum(labels == 1))
    spaces = {
        "Logistic Regression": {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        "Decision Tree": {
            "max_depth": [6, 10, 14],
            "min_samples_split": [20, 60],
            "min_samples_leaf": [10, 30, 60],
            "class_weight": [None, "balanced"],
        },
        "KNN": {
            "n_neighbors": [5, 15, 31],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Random Forest": {
            "n_estimators": [100, 160],
            "max_depth": [12, 18],
            "min_samples_leaf": [2, 8],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced_subsample"],
        },
        "XGBoost": {
            "n_estimators": [200, 350],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 5],
            "scale_pos_weight": [1.0, ratio],
        },
    }
    defaults = {
        "Logistic Regression": {"C": 1.0, "class_weight": None},
        "Decision Tree": {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 30,
            "class_weight": None,
        },
        "KNN": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        "Random Forest": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "XGBoost": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "scale_pos_weight": 1.0,
        },
    }
    first = defaults[algorithm]
    second = {
        **first,
        **(
            {"weights": "uniform", "n_neighbors": 15}
            if algorithm == "KNN"
            else {"scale_pos_weight": ratio}
            if algorithm == "XGBoost"
            else {
                "class_weight": "balanced_subsample" if algorithm == "Random Forest" else "balanced"
            }
        ),
    }
    candidates = [first, second]
    sample_count = min(32, int(np.prod([len(v) for v in spaces[algorithm].values()])))
    for value in ParameterSampler(spaces[algorithm], n_iter=sample_count, random_state=config.seed):
        if value not in candidates:
            candidates.append(value)
        if len(candidates) >= config.search_candidates:
            break
    return candidates[: config.search_candidates]


def positive_probability(estimator, matrix):
    classes = np.asarray(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Fitted classifiers must have both explicit synthetic classes")
    result = np.asarray(
        estimator.predict_proba(matrix)[:, int(np.flatnonzero(classes == 1)[0])], dtype=float
    )
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Classifier emitted invalid fraud probabilities")
    return result


def benchmark_knn(train, labels, validation, validation_labels, config):
    rows = representative_rows(labels, min(config.knn_sample_size, len(labels)), config.seed)
    queries = representative_rows(
        validation_labels, min(config.knn_benchmark_queries, len(validation_labels)), config.seed
    )
    estimator = make_estimator("KNN", search_parameters("KNN", labels, config)[0], config)
    start = perf_counter()
    estimator.fit(train[rows], labels[rows])
    fit_time = perf_counter() - start
    start = perf_counter()
    p = positive_probability(estimator, validation[queries])
    prediction_time = perf_counter() - start
    full_pairs = len(labels) * len(validation_labels)
    projected = prediction_time * full_pairs / (len(rows) * len(queries))
    constrained = full_pairs > config.knn_max_pairs or projected > config.knn_max_projected_seconds
    report = KNNBenchmark(
        sample_size=len(rows),
        fraud_count=int(labels[rows].sum()),
        query_count=len(queries),
        fit_seconds=fit_time,
        prediction_seconds=prediction_time,
        matrix_memory_bytes=matrix_summary(train[rows])["memory_bytes"],
        full_pair_count=full_pairs,
        projected_full_prediction_seconds=projected,
        constrained=constrained,
        reason="Representative stratified training subset; full fit/prediction "
        "exceeds configured pair/time budget"
        if constrained
        else "Measured benchmark and configured pair/time budgets permit full training",
        metrics=metrics(validation_labels[queries], p),
    )
    return rows if constrained else np.arange(len(labels)), report
