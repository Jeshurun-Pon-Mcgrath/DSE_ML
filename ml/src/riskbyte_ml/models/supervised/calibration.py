"""Calibration learns earlier validation labels; choices use later validation only."""

from time import perf_counter

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from riskbyte_ml.models.supervised.metrics import metrics
from riskbyte_ml.models.supervised.schemas import CalibrationChoice


class ProbabilityCalibrator:
    def __init__(self, method="none"):
        self.method = method
        self.fitted = method == "none"
        self.estimator = None

    @staticmethod
    def logits(probabilities):
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probabilities, labels):
        if self.fitted:
            raise ValueError("Calibrators cannot be refitted")
        if self.method == "sigmoid":
            self.estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            self.estimator.fit(self.logits(probabilities), labels)
        elif self.method == "isotonic":
            self.estimator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            self.estimator.fit(probabilities, labels)
        else:
            raise ValueError("Unsupported calibration method")
        self.fitted = True
        return self

    def transform(self, probabilities):
        if not self.fitted:
            raise ValueError("Calibrator must be fitted")
        if self.method == "none":
            return np.asarray(probabilities, dtype=float)
        if self.method == "sigmoid":
            return self.estimator.predict_proba(self.logits(probabilities))[:, 1]
        return self.estimator.predict(probabilities)


def select_calibration(fit_probabilities, fit_labels, selection_probabilities, selection_labels):
    start = perf_counter()
    calibrators = {"none": ProbabilityCalibrator()}
    excluded = {}
    for method in ("sigmoid", "isotonic"):
        if len(np.unique(fit_labels)) < 2:
            excluded[method] = "calibration fitting block lacks both classes"
        elif method == "isotonic" and (len(fit_labels) < 1000 or np.sum(fit_labels) < 100):
            excluded[method] = (
                "isotonic requires >=1000 fitting rows and >=100 fraud rows in this protocol"
            )
        else:
            calibrators[method] = ProbabilityCalibrator(method).fit(fit_probabilities, fit_labels)
    evaluated = {
        name: metrics(selection_labels, value.transform(selection_probabilities))
        for name, value in calibrators.items()
    }
    baseline = evaluated["none"]
    # Brier improvement must exceed 1%, without AP deterioration beyond 0.01.
    eligible = [
        name
        for name, m in evaluated.items()
        if name != "none"
        and m.brier_score < baseline.brier_score * 0.99
        and m.pr_auc is not None
        and baseline.pr_auc is not None
        and m.pr_auc >= baseline.pr_auc - 0.01
    ]
    selected = min(eligible, key=lambda name: evaluated[name].brier_score) if eligible else "none"
    report = CalibrationChoice(
        method=selected,
        fitting_source="earlier chronological validation block; never training/test refit",
        selection_source="later chronological validation block",
        candidates=evaluated,
        excluded_methods=excluded,
        reasoning="Choose minimum selection-block Brier only with >1% improvement "
        "over raw and AP loss <=0.01; otherwise retain raw",
    )
    return calibrators[selected], report, perf_counter() - start
