"""Measured, report-ready tables and standalone vector curves from saved evaluations."""

import csv
import html
import io
import json

from riskbyte_ml.data.artifacts import write_artifact


def fmt(value):
    return "unavailable" if value is None else f"{value:.6f}"


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ]


def write_csv(path, headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    write_artifact(path, stream.getvalue().encode())


def curve_svg(path, series, title, xlabel, ylabel):
    """Exact recorded display points, explicit fixed axes; no invented chart values."""
    colours = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#475569"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" '
        'viewBox="0 0 900 520" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="900" height="520" fill="white"/>',
        f'<text x="60" y="28" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for index in range(6):
        x, y = 60 + index * 104, 440 - index * 78
        parts.extend(
            [
                f'<path d="M{x} 50V440 M60 {y}H580" stroke="#e2e8f0" fill="none"/>',
                f'<text x="{x}" y="460" font-size="12">{index / 5:.1f}</text>',
                f'<text x="28" y="{y}" font-size="12">{index / 5:.1f}</text>',
            ]
        )
    for index, (name, points) in enumerate(series.items()):
        color = colours[index % len(colours)]
        coordinates = " ".join(f"{60 + p['x'] * 520:.3f},{440 - p['y'] * 390:.3f}" for p in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="610" y="{80 + index * 30}" fill="{color}" '
            f'font-family="sans-serif" font-size="13">{html.escape(name)}</text>'
        )
    parts.extend(
        [
            f'<text x="250" y="498" font-size="14">{html.escape(xlabel)}</text>',
            '<text transform="translate(15 280) rotate(-90)" font-size="14">'
            f"{html.escape(ylabel)}</text>",
            "</svg>",
        ]
    )
    write_artifact(path, "\n".join(parts).encode())


def generate_report(registry, output):
    directory = output / "evaluation"
    selected = next(m for m in registry.models if m.is_selected_supervised_model)
    headers = ["Model", "Precision", "Recall", "F1", "PR-AUC (AP)", "ROC-AUC", "FPR", "FNR"]

    def comparison_rows(source):
        rows = []
        for model in registry.models:
            value = getattr(model, source)
            m = value.metrics if value else None
            rows.append(
                [
                    model.algorithm,
                    *(
                        fmt(getattr(m, n)) if m else "unavailable"
                        for n in ("precision", "recall", "f1", "pr_auc", "roc_auc", "fpr", "fnr")
                    ),
                ]
            )
        return rows

    primary = comparison_rows("selection")
    full = comparison_rows("validation")
    hard_headers = ["Model", "Hard negatives", "False positives", "False-positive rate"]
    hard_rows = [
        [
            m.algorithm,
            m.validation.hard_negatives.count if m.validation else "unavailable",
            m.validation.hard_negatives.flagged if m.validation else "unavailable",
            fmt(m.validation.hard_negatives.rate) if m.validation else "unavailable",
        ]
        for m in registry.models
    ]
    names = list(selected.validation.scenario_recall)
    scenario_headers = ["Scenario", *(m.algorithm for m in registry.models)]
    scenario_rows = [
        [
            name,
            *(
                fmt(m.validation.scenario_recall[name].rate) if m.validation else "unavailable"
                for m in registry.models
            ),
        ]
        for name in names
    ]
    timing_headers = [
        "Model",
        "Search fit seconds",
        "Chosen fit seconds",
        "Validation predict seconds",
    ]
    timing_rows = [
        [
            m.algorithm,
            fmt(m.timing.search_training_seconds) if m.timing else "not fitted",
            fmt(m.timing.chosen_fit_seconds) if m.timing else "not fitted",
            fmt(m.validation.prediction_seconds) if m.validation else "unavailable",
        ]
        for m in registry.models
    ]
    for name, columns, rows in [
        ("model_comparison", headers, primary),
        ("whole_validation_comparison", headers, full),
        ("hard_negative_performance", hard_headers, hard_rows),
        ("scenario_recall", scenario_headers, scenario_rows),
        ("timing", timing_headers, timing_rows),
    ]:
        write_csv(directory / (name + ".csv"), columns, rows)
    curve_series = {"pr": {}, "roc": {}}
    for model in registry.models:
        path = directory / (model.model_id + "_curves.json")
        if path.is_file():
            value = json.loads(path.read_bytes())
            for kind in curve_series:
                curve_series[kind][model.algorithm] = value[kind]
    curve_svg(
        directory / "validation_pr.svg",
        curve_series["pr"],
        "Synthetic-data finding: whole validation Precision–Recall",
        "Recall",
        "Precision",
    )
    curve_svg(
        directory / "validation_roc.svg",
        curve_series["roc"],
        "Synthetic-data finding: whole validation ROC",
        "False-positive rate",
        "True-positive rate",
    )
    test_points = registry.selected_test_curves.model_dump(mode="json")
    for kind in ("pr", "roc"):
        curve_svg(
            directory / ("selected_test_" + kind + ".svg"),
            {selected.algorithm: test_points[kind]},
            f"Synthetic-data finding: frozen selected classifier test {kind.upper()}",
            "Recall" if kind == "pr" else "False-positive rate",
            "Precision" if kind == "pr" else "True-positive rate",
        )
    lines = [
        "# Supervised model evaluation — Synthetic-data findings",
        "",
        "All measurements describe the RiskByte Synthetic Fraud Dataset: an IEEE-CIS-inspired "
        "academic simulation. They do not establish banking performance, causal explanations or "
        "production-ready calibrated probabilities.",
        "",
        f"Dataset fingerprint: `{registry.provenance.dataset_fingerprint}`.",
        f"Feature version: `{registry.provenance.feature_version}`.",
        f"Primary temporal split: `{registry.provenance.split_id}`.",
        f"Frozen feature manifest content hash: `{registry.provenance.feature_manifest_hash}`.",
        f"Model version: `{registry.provenance.model_version}`; "
        f"registry version: `{registry.provenance.registry_version}`.",
        f"Experiment ID: `{registry.provenance.run_id}`.",
        f"Generated UTC: `{registry.provenance.generated_at.isoformat()}`.",
        "",
        "## Method and isolation contract",
        "",
        "Existing verified train-only "
        "RobustScaler/median/rare-grouping/frequency/OHE "
        "preprocessing "
        "serves Logistic Regression/KNN. Unscaled tree preprocessing serves Decision Tree, Random "
        "Forest and optional XGBoost. Both frozen CSR families retain 144 encoded columns. No "
        "dataset, split, feature manifest or temporal preprocessor is modified/refitted.",
        "",
        "Models fit only the 70,000 training rows. Earlier chronological validation rows select "
        "restrained hyperparameters and fit calibrators; later validation rows select calibration "
        "method, operating threshold and final algorithm. Internal validation blocks do not change "
        "the frozen split. No random cross-validation or preprocessing across temporal folds. "
        "ParameterSampler generates a small reproducible "
        "fixed-validation search, including default "
        "and weighted behaviour. XGBoost uses only the earlier validation block for early stopping "
        "when installed. Nonconverged candidates are ineligible. No SMOTE or target encoding.",
        "",
        f"Validation block counts: `{registry.validation_blocks}`. "
        f"Dependency versions: `{registry.provenance.dependency_versions}`.",
        "",
        "Sigmoid calibration fits clipped-logit scores on earlier validation; isotonic requires "
        "at least 1,000 fitting rows and 100 fraud observations. On later validation, choose the "
        "lowest Brier alternative only if it improves raw Brier by >1% without AP loss >0.01. "
        "Otherwise retain raw probabilities. Reliability bins and Brier scores expose empirical "
        "probability quality; they cannot validate real-world banking probabilities.",
        "",
        "Default BALANCED policy maximizes validation F1 subject to FPR <=1% and hard-negative "
        "FPR <=10%; ties prefer recall then precision. HIGH_RECALL maximizes recall with FPR <=2% "
        "and hard-negative FPR <=15%. LOW_FALSE_POSITIVE minimizes FPR with recall >=60% and "
        "hard-negative FPR <=5%. Policies are configurable simulation criteria, not bank-approved "
        "cost thresholds. All exact distinct score candidates are "
        "considered; charts retain bounded "
        "points. Decision is P(fraud) >= threshold; >1 is an "
        "explicit all-negative endpoint/fallback.",
        "",
        "PR-AUC is non-interpolated average precision, computed from full predictions, not a "
        "trapezoid through sampled display points. Low prevalence makes accuracy and ROC-AUC alone "
        "poor selection criteria. A false positive flags a "
        "legitimate transaction; a false negative "
        "misses a simulated attack. Hard-negative and scenario "
        "counts are evaluation metadata only.",
        "",
        "## Primary later-validation comparison",
        "",
        "These rows support selection. Calibration method and threshold use this block, so even "
        "these validation results contain selection optimism. Accuracy is secondary.",
        "",
        *table(headers, primary),
        "",
        "## Entire-validation comparison",
        "",
        "Includes earlier calibration-fitting rows; not an independent generalization estimate.",
        "",
        *table(headers, full),
        "",
        "## Operating points, calibration and imbalance",
        "",
        *table(
            ["Model", "Threshold", "Calibration", "Selection Brier", "Imbalance strategy"],
            [
                [
                    m.algorithm,
                    fmt(m.selected_threshold),
                    m.calibration.method if m.calibration else "unavailable",
                    fmt(m.selection.metrics.brier_score) if m.selection else "unavailable",
                    m.class_imbalance_strategy,
                ]
                for m in registry.models
            ],
        ),
        "",
        "Every trained model also records its 0.5-threshold metrics, selected train diagnostics, "
        "candidate parameters/results, full threshold-policy curves and tree global importances.",
        "",
        "## Hard negatives — entire validation",
        "",
        *table(hard_headers, hard_rows),
        "",
        "Each subtype is reported independently in the JSON registry, with explicit unavailable "
        "rates for empty subgroups. Unusual legitimate purchases must not be interpreted as fraud "
        "solely because an amount/device/velocity is unusual.",
        "",
        "## Scenario recall — entire validation",
        "",
        *table(scenario_headers, scenario_rows),
        "",
        "Counts/true positives accompany every rate in the registry. "
        "Small subgroups have uncertain "
        "recall; evaluation-only fraud_scenario never enters matrices or rule inputs.",
        "",
        "## Selection and final temporal test",
        "",
        registry.selection_freeze.rationale,
        "",
        "Selection, calibrated artifact hash and operating threshold were exclusively published "
        "before the one-use test gate opened. Only the selected supervised classifier was scored "
        "on the primary temporal test. No refitting/tuning followed. Other models' test metrics "
        "remain unavailable. Phase 3 had already inspected aggregate test statistics; this is "
        "Phase 4 model-selection isolation, not a claim that prior "
        "exploration never saw test data.",
        "",
        *table(
            ["Partition", "Rows", "Fraud prevalence"],
            [
                [n, registry.partition_counts[n], fmt(registry.partition_prevalence[n])]
                for n in ("train", "validation", "test")
            ],
        ),
        "",
        *table(
            ["Scope", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "FPR", "FNR", "Brier"],
            [
                [
                    name,
                    *(
                        fmt(getattr(v.metrics, n))
                        for n in (
                            "precision",
                            "recall",
                            "f1",
                            "pr_auc",
                            "roc_auc",
                            "fpr",
                            "fnr",
                            "brier_score",
                        )
                    ),
                ]
                for name, v in [
                    ("selection", selected.selection),
                    ("whole validation", selected.validation),
                    ("final test", selected.test_metrics),
                ]
            ],
        ),
        "",
        f"Final test confusion: `{selected.test_metrics.metrics.confusion.model_dump()}`.",
        f"Final test hard negatives: `{selected.test_metrics.hard_negatives.model_dump()}`.",
        "",
        *table(
            ["Scenario", "Test fraud count", "Detected", "Recall"],
            [
                [name, s.count, s.flagged, fmt(s.rate)]
                for name, s in selected.test_metrics.scenario_recall.items()
            ],
        ),
        "",
        "## Error analysis — selected final test",
        "",
        f"Definitions: {registry.selected_test_errors.threshold_definitions}.",
        "",
        f"False positives: `{registry.selected_test_errors.false_positives.model_dump()}`.",
        "",
        f"False negatives: `{registry.selected_test_errors.false_negatives.model_dump()}`.",
        "",
        "These are aggregate diagnostics, not exposed individual transactions. Investigate "
        "hard-negative/new-device/high-amount/velocity false "
        "positives and weak-signal or missing-identity "
        "false negatives without adjusting this already frozen model.",
        "",
        "## Timing and KNN practicality",
        "",
        *table(timing_headers, timing_rows),
        "",
        f"Selected test batch prediction: {fmt(selected.test_metrics.prediction_seconds)} seconds; "
        f"average {fmt(selected.test_metrics.seconds_per_transaction * 1e6)} "
        "microseconds/transaction. "
        "This excludes preprocessing, history, HTTP and queueing, and is not a real-time SLA.",
        "",
        "Random Forest uses serial tree aggregation for exact serialization/probability tie "
        "reproducibility. Other estimators obey the bounded thread/working-memory budget.",
        "",
    ]
    for model in registry.models:
        if model.knn_benchmark:
            b = model.knn_benchmark
            lines.extend(
                [
                    f"KNN benchmark: `{b.model_dump(mode='json')}`.",
                    "",
                    "Projected full timing is an explicit extrapolation from the measured sample, "
                    "not a completed MEDIUM KNN run. Retained CSR bytes and a 64 "
                    "MiB distance working "
                    "budget describe memory behaviour, not isolated peak RSS. Constrained train "
                    "diagnostics use the fit subset; KNN stores that subset's "
                    "encoded vectors by design.",
                    "",
                ]
            )
        if model.limitation:
            lines.extend([f"{model.algorithm}: {model.limitation}", ""])
    lines.extend(
        [
            "## Secondary saved random-split benchmark",
            "",
            "Run only after the primary selection/test freeze. Strong "
            "available primary models reuse "
            "chosen hyperparameters, with new train-only random-split "
            "preprocessing saved separately. "
            "Only random validation is scored at raw threshold 0.5; random test is unscored. AP is "
            "more comparable than differently calibrated operating-point "
            "rates. Random training reuses "
            "some primary test rows; these results cannot inform any primary model decision. Prior "
            "unlabelled history is transductive, so this is not "
            "independent future-period evaluation.",
            "",
            *table(
                ["Model", "Temporal selection AP", "Random validation AP", "Random fit seconds"],
                [
                    [
                        r.algorithm,
                        fmt(r.temporal_selection_pr_auc),
                        fmt(r.validation.metrics.pr_auc),
                        fmt(r.training_seconds),
                    ]
                    for r in registry.random_comparisons
                ],
            ),
            "",
            "## Phase 3 connection and feature importance",
            "",
            "Phase 3 found strong velocity separation and weak amount-ratio separation between "
            "fraud and hard negatives. The measured model allocations below permit reviewing "
            "consistency; correlations/importance do not demonstrate causal reasons or guarantee "
            "false-positive improvement. SHAP is not implemented. Anonymized features would retain "
            "ANONYMIZED tags; the frozen generator enables no V features.",
            "",
            *table(
                ["Selected encoded feature", "Global importance", "Interpretability"],
                [
                    [f.feature, fmt(f.importance), f.interpretability_class]
                    for f in selected.feature_importance[:15]
                ],
            ),
            "",
            "## Curves and reusable artifacts",
            "",
            "Local reports: "
            "`ml/artifacts/evaluation/model_comparison.json`, five "
            "report-ready CSV tables, "
            "per-model validation PR/ROC/reliability JSON, "
            "final_test_metrics.json and standalone SVGs. "
            "PR/ROC charts use recorded finite curve points; AUC uses "
            "all scored records. Whole-validation "
            "curves include calibration-fitting rows. Models/metadata "
            "are under `ml/artifacts/models/`; "
            "only trusted, checksummed local joblib artifacts may be loaded. API reads JSON only.",
            "",
            "![Whole-validation PR](../ml/artifacts/evaluation/validation_pr.svg)",
            "",
            "![Whole-validation ROC](../ml/artifacts/evaluation/validation_roc.svg)",
            "",
            "![Selected final-test PR](../ml/artifacts/evaluation/selected_test_pr.svg)",
            "",
            "## Leakage audit",
            "",
            *(f"- {name}: {result}" for name, result in registry.leakage_audit.items()),
            "",
            "## Reproduce and inspect",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m riskbyte_ml.models.supervised "
            "--config ml/config/synthetic_riskbyte.example.json "
            "--experiment-config ml/config/supervised.example.json",
            "```",
            "",
            "A completed identical experiment returns the saved registry "
            "without refitting or reopening "
            "the test gate. Changed settings/dependencies are rejected in that output directory. A "
            "frozen/interrupted experiment cannot silently retrain or "
            "repeat the test. Use one offline "
            "writer; fresh output directories do not make an already "
            "inspected test scientifically unseen.",
            "",
            "Read-only routes: GET /api/v1/models, /comparison, /selected, /{model_id}/metrics, "
            "/{model_id}/scenario-recall, "
            "/{model_id}/threshold-analysis. No public prediction API, "
            "startup training or frontend redesign.",
            "",
            "Optional XGBoost setup in an unrestricted environment: "
            '`python -m pip install -e "./ml[boosting]"`. Adding a missing algorithm after final '
            "evaluation requires a separately declared experiment; do "
            "not silently reuse the primary "
            "test as untouched or change this frozen selected classifier.",
            "",
            "## Limitations",
            "",
            *("- " + item for item in registry.limitations),
            "",
            "## References",
            "",
            "[Average "
            "precision](https://scikit-learn.org/stable/modules/generated"
            "/sklearn.metrics.average_precision_score.html), "
            "[calibration](https://scikit-learn.org/stable/modules/calibration.html), "
            "[XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).",
            "",
            "## Next recommended phase",
            "",
            "Review the selected classifier, hard-negative/scenario "
            "tradeoffs and unavailable dependency. "
            "Then separately authorize Phase 5 anomaly detection with "
            "training-only reference populations "
            "and score normalization. Do not fuse scores, implement "
            "SHAP/ABRF or deploy based on this "
            "synthetic benchmark. No Phase 5 work is performed here.",
            "",
        ]
    )
    report = "\n".join(lines)
    write_artifact(directory / "SUPERVISED_MODEL_EVALUATION.md", report.encode())
    return report
