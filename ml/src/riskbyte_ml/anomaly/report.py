"""Reproducible Markdown reporting from measured development artifacts only."""

from riskbyte_ml.data.artifacts import write_artifact


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        ]
    )


def write_reports(bundle, directory):
    value = bundle.model_dump(mode="json")
    selected, metrics = value["selected"], value["anomaly_metrics"]
    groups = value["distributions"]["groups"]
    base = value["complementarity"]["supervised_validation"]
    provenance = (
        f"**Synthetic-data finding.** Dataset `{bundle.dataset_fingerprint}`; feature pipeline "
        f"`{bundle.feature_version}`; temporal split `{bundle.split_id}`. "
        "Development contains 70,000 TRAIN and 15,000 VALIDATION rows. "
        "The original temporal test partition was not decoded into records, scored, or used "
        "for Phase 5 decisions. Full files are read as opaque bytes for checksums; identity "
        "projection checks excluded numeric join keys without decoding excluded context fields.\n\n"
    )
    candidate_table = table(
        [
            "Features",
            "Reference",
            "Raw/prepared columns",
            "AP",
            "ROC-AUC",
            "Recall",
            "Legitimate FPR",
            "Hard-negative FPR",
            "Fit seconds",
        ],
        [
            [
                x["family"],
                x["population"],
                f"{x['raw_feature_count']}/{x['prepared_feature_count']}",
                f"{x['validation']['pr_auc']:.6f}",
                f"{x['validation']['roc_auc']:.6f}",
                f"{x['validation']['recall']:.4%}",
                f"{x['validation']['fpr']:.4%}",
                f"{x['validation']['hard_negatives']['rate']:.4%}",
                f"{x['fit_seconds']:.3f}",
            ]
            for x in value["study"]
        ],
    )
    anomaly = (
        "# Phase 5 anomaly detection\n\n"
        + provenance
        + f"""## Method and selection

Isolation Forest isolates unusual observations using randomly partitioned trees. Shorter
average isolation paths indicate global unusualness, not proof of fraud. Version:
`{bundle.anomaly_version}`. Four candidates compare interpretable versus broader numeric
features and legitimate-only versus mixed training. No categorical one-hot matrix is fed
to the detector. The broader contract excludes identifiers, categories and absolute elapsed
position to avoid identifying future time as automatically anomalous.

The reference contract deliberately prefers known legitimate TRAIN observations; mixed
training is a methodological comparator. TRAIN labels select the normal reference cohort,
but fitting receives X only. Validation labels evaluate fitted candidates and thresholds.
No outcome/scenario/hard-negative field enters X. The selected candidate has the highest
validation average precision among eligible legitimate-reference candidates.

{candidate_table}

Selected: **{selected["family"]}**, **{selected["population"]}**, **{selected["fit_rows"]:,}**
reference rows; **{selected["raw_feature_count"]}** numeric signals plus one missing indicator
per signal (**{selected["prepared_feature_count"]}** prepared columns). Seed 42, 150 trees,
maximum sample 1,024 per tree. These are restrained prespecified settings, not a giant search.

Selected fields: {", ".join("`" + x + "`" for x in selected["features"])}.

## Missingness and score semantics

Each candidate fits medians on its TRAIN reference only. All-missing columns use zero plus
an explicit missing indicator. Nonfinite values become missing; numerical preparation uses
the existing documented fixed ±1e12 safety bound. No source rows/columns are deleted.
No scaling or one-hot expansion is necessary for this numeric isolation study.

The original `score_samples` output is preserved, with lower values more anomalous.
Raw anomaly severity is its negative. A strict-left empirical CDF of TRAIN reference severity
maps severity to [0,1]: `searchsorted(sorted_training_scores, x, side='left') / n`.
This deterministic monotonic mapping is frozen before validation. Values beyond the training
reference clip naturally to 0/1; ties share a percentile. **A is not fraud probability.**

Contamination candidates: auto, .005, .01, .02, .03, .05. For fixed trees, contamination
changes the native decision offset, not tree fitting/ranking. Thus offsets are evaluated
without redundantly fitting six identical forests. Numeric offsets use TRAIN-reference
quantiles; auto uses raw severity 0.5. Native decisions use strict raw comparison.
Chosen native contamination: **{selected["contamination"]}**. This is not an estimate of
fraud prevalence. The operational analysis threshold is chosen separately on normalized A.

## Validation operating point

Threshold **{metrics["threshold"]:.16f}** maximizes fraud recall subject to overall legitimate
FPR ≤2% and overall hard-negative rate ≤10%; ties favor precision, fewer FPs, higher threshold.
Continuous scores remain primary. Caps constrain overall hard negatives, not every subtype.
Raw threshold points and every contamination comparison remain in `analysis_bundle.json`.

AP **{metrics["pr_auc"]:.6f}**, ROC-AUC **{metrics["roc_auc"]:.6f}**, precision
**{metrics["precision"]:.4%}**, recall **{metrics["recall"]:.4%}**, F1 **{metrics["f1"]:.6f}**,
legitimate FPR **{metrics["fpr"]:.4%}**, hard-negative rate
**{metrics["hard_negatives"]["rate"]:.4%}**
({metrics["hard_negatives"]["flagged"]}/{metrics["hard_negatives"]["count"]}).
Confusion: {metrics["confusion"]}. AP means non-interpolated average precision.

## Artifacts, API and execution

`ml/artifacts/phase5/isolation_forest.joblib` contains the trusted fitted model, feature list,
imputer, sorted normalization reference and provenance. `anomaly_metadata.json` records
population, contamination, feature study and validation threshold; the coherent
`analysis_bundle.json` binds versions, seed/configuration, split, timestamp, baseline hash
and executable artifact hashes. No full raw dataset is serialized.
GET `/api/v1/anomaly/{{status,metrics,scenarios,hard-negatives}}` reads JSON/checksums only.
No prediction endpoint, model fitting, or unpickling occurs in API requests.

```powershell
.\\.venv\\Scripts\\python.exe -m riskbyte_ml.anomaly `
  --config ml/config/synthetic_riskbyte.example.json `
  --experiment-config ml/config/anomaly.example.json `
  --behaviour-config ml/config/behaviour.example.json
```

Completed identical runs reuse verified artifacts; changed configuration or incompatible
frozen sources abort. Executable joblib artifacts are trusted local files only.

## Limitations

Synthetic academic data cannot establish production banking performance. Generator rules
influence separability. Normal training contains unusual legitimate hard negatives; anomaly
is not equivalent to fraud. Validation is reused for feature-family and threshold selection,
so its measurements are development results. Episode/entity dependence and small scenario
counts limit uncertainty claims. Empirical percentiles may shift under drift; they are not
calibrated probabilities. Numerical preparation retains a modest dense matrix and no OHE.
No original-test candidate evaluation, ABRF, SHAP or final system logic is implemented.
"""
    )
    behavioural_rows = [
        [
            name,
            f"{data['behaviour_score']['mean']:.6f}",
            f"{data['behaviour_score']['median']:.6f}",
            f"{data['behaviour_confidence']['mean']:.6f}",
            f"{data['behaviour_flag_rate']['rate']:.4%}",
        ]
        for name, data in groups.items()
    ]
    behaviour = (
        "# Phase 5 behavioural deviation\n\n"
        + provenance
        + f"""## Entity baseline and strict-prior evidence

Version `{bundle.behaviour_version}` is a deterministic engine, separate from Isolation
Forest and the supervised classifier. The unchanged frozen feature builder uses only
earlier event times for each simulated behavioural entity. Equal-time peers do not enter
each other's history. Prior labels never filter entity history; earlier attack observations
can contaminate baselines. Synthetic entities are not actual bank customers.

## Components and provisional configuration

Amount ratio severity is `abs(log(max(ratio, 1e-6)))` for nonnegative ratios. This considers
both decreases and increases. Z-score severity is its absolute value. Velocity uses 30-minute
and 24-hour prior counts. Short-gap severity is `1/(1+prior_gap_seconds)`. Failed attempts
are pre-outcome counts. Each continuous severity uses a strict-left empirical CDF fitted
on ALL TRAIN features, without target labels. Zero-mass/minimum components map to zero.
Merchant/device/billing-region/product novelty is `1 - seen_before`, but only where prior
context exists. Unknown context stays unavailable, never automatically novel or safe.

Ten components have explicit equal provisional weights in `ml/config/behaviour.example.json`.
Amount ratio/z-score and correlated velocity/gap components are not independent; equal weights
are a transparent starting definition, not calibrated risk contributions. Weights must ultimately
be validated/calibrated experimentally. No label-driven weight search occurs in Phase 5.
B is the weighted mean of AVAILABLE components. Missing components are excluded explicitly
from the denominator; coverage is reported separately. Individual contributions sum to B
when overall history is available. Structured explanations expose raw values, normalized
severity, contribution, direction, availability and window; no fabricated sample values.

## Confidence and cold start

Confidence is `n/(n+20) * available_weight_fraction`, where n is strictly prior transaction
count. Saturation 20 and minimum history 1 are configurable reliability assumptions, not
empirical fraud probabilities. Context missingness (including absent device evidence) reduces
coverage. History duration is explicitly NOT used because frozen features lack a prior-span
field. Future research should assess duration and independent context coverage before calling
confidence calibrated reliability.

First entity event: history_count=0, history_available=false, B=null, confidence=0.
Missing history is not zero risk. All unavailable scores are treated as abstentions; reports
retain full-row denominators for flag rates and separately disclose available-score AUC scope.
`DeviationEngine.explain` returns typed component records. The API exposes aggregate reports,
not raw development transactions.

## Measured validation findings

{table(["Cohort", "Mean B", "Median B", "Confidence mean", "B flag rate"], behavioural_rows)}

The analysis-only B threshold is **{value["behaviour_metrics"]["threshold"]:.16f}**, selected
under the same validation false-positive caps. Recall is
**{value["behaviour_metrics"]["recall"]:.4%}**; legitimate flag rate
**{value["behaviour_metrics"]["fpr"]:.4%}**; hard-negative flag rate
**{value["behaviour_metrics"]["hard_negatives"]["rate"]:.4%}**.
No B threshold is deployed as a banking fraud decision.
**B and confidence are not fraud probabilities.**

## Availability and limitations

International novelty is unavailable: the frozen feature contract lacks prior international
history. The anomaly engine can use raw is_international, while B does not silently relabel
it as entity-specific novelty. No GPS/distance feature exists. Time gap is global training
gap severity, not a learned per-entity normal-gap model. Confidence currently measures count
and component coverage, not prior duration or independently calibrated reliability.
Missing device identity, sparse baselines, correlated components and contaminated prior
history can limit evidence. Synthetic findings cannot prove production performance.

`behavioural_engine.joblib` stores only configuration and fitted references/provenance;
`behavioural_metadata.json` documents sources, definitions and confidence. The evaluation
sidecar `development_scores.csv` keeps P/A/B, confidence and outcome metadata OUTSIDE X.
GET `/api/v1/behaviour/{{summary,components,validation-analysis}}` serves read-only summaries.
No score fusion, ABRF, SHAP or frontend redesign occurs.
"""
    )
    scenario_table = table(
        ["Fraud scenario", "N", "Supervised recall", "Anomaly recall", "Median B"],
        [
            [
                name,
                x["count"],
                f"{base['scenario_recall'][name]['rate']:.4%}" if x["count"] else "unavailable",
                f"{x['rate']:.4%}" if x["count"] else "unavailable",
                value["distributions"]["scenarios"][name]["behaviour_score"]["median"],
            ]
            for name, x in metrics["scenario_recall"].items()
        ],
    )
    hard_table = table(
        [
            "Legitimate hard-negative type",
            "N",
            "Anomaly flags",
            "Anomaly rate",
            "Behaviour flags",
            "Behaviour rate",
        ],
        [
            [
                name,
                x["count"],
                x["flagged"],
                f"{x['rate']:.4%}" if x["count"] else "unavailable",
                value["behaviour_metrics"]["hard_negative_types"][name]["flagged"],
                value["behaviour_metrics"]["hard_negative_types"][name]["rate"],
            ]
            for name, x in metrics["hard_negative_types"].items()
        ],
    )
    complementary = value["complementarity"]["validation"]
    focused = []
    for name in ["VELOCITY_BURST", "LOW_AND_SLOW", "INTERNATIONAL_ANOMALY"]:
        a = metrics["scenario_recall"][name]
        p = base["scenario_recall"][name]
        effects = value["investigations"][name]
        median = value["distributions"]["scenarios"][name]["behaviour_score"]["median"]
        focused.append(
            f"**{name}**: anomaly flags {a['flagged']}/{a['count']}; supervised flags "
            f"{p['flagged']}/{p['count']}. Anomaly-score Cliff's delta versus normal legitimate "
            f"is {effects['anomaly_score']['cliffs_delta_vs_normal']:.6f}; B delta is "
            f"{effects['behaviour_score']['cliffs_delta_vs_normal']:.6f}. Median B is "
            f"{median:.6f} (analysis threshold {value['behaviour_metrics']['threshold']:.6f})."
        )
    focused_text = "\n\n".join(focused)
    complement = (
        "# Phase 5 supervised complementarity\n\n"
        + provenance
        + f"""## Frozen supervised reference

Decision Tree `{value["baseline"]["model_id"]}` remains byte-identical, with unchanged
sigmoid calibration and threshold `{value["baseline"]["threshold"]}`. Phase 4 registry,
selection history and test results are never rewritten. Existing test blind spots motivate
research questions only; no old test records inform new thresholds or normalization.
P is the supervised fraud probability; A global anomaly and B entity deviation are separate.

## What anomaly detection recovers

Validation fraud contingency: supervised only **{complementary["supervised_only"]}**, anomaly
only **{complementary["anomaly_only"]}**, both **{complementary["both"]}**, neither
**{complementary["neither"]}**. The classifier detects **{complementary["supervised_detected"]}**
of 266 fraud rows. Analysis-only OR covers **{complementary["analytical_or_detected"]}**,
recovering **{complementary["additional_fraud_recovered"]}** additional fraud rows while adding
**{complementary["additional_false_positives"]}** false positives. This tradeoff does NOT justify
deploying an OR rule. No operational ensemble/fusion logic exists.

{scenario_table}

## Dedicated validation investigations

{focused_text}

Ranking separation does not imply adequate threshold recall under hard-negative constraints.
Longer-window and relative amount evidence remains available for later frozen-component
ablations, with no special scenario-aware fraud rule. Global unusualness can overlap hard
negatives. Travel hard negatives are separately retained; international activity is not
fraud and is not a documented entity-specific international novelty signal.

All focused raw-signal and A/B quantiles/effects are saved under `investigations` in the bundle.
These are descriptive synthetic effects without independent population inference.

## Hard-negative protection and overlap

{hard_table}

Hard-negative A median **{groups["hard_negative"]["anomaly_score"]["median"]:.6f}** overlaps
fraud A median **{groups["fraud"]["anomaly_score"]["median"]:.6f}**; normal legitimate A median
is **{groups["normal_legitimate"]["anomaly_score"]["median"]:.6f}**. B cohort distributions and
confidence/quantiles appear in the behavioural report. Large/shopping-burst legitimate
purchases can resemble attacks. Overall 10% hard-negative caps do not ensure subtype parity.

## Artifacts and audit

`ml/artifacts/phase5/` contains two executable engine objects, anomaly/behaviour metadata,
coherent analysis bundle, complementarity and leakage-audit JSON, and an 85,000-row
evaluation-only development score sidecar. Separate TRAIN diagnostics use validation-frozen
thresholds and are not unbiased generalization estimates. All source/artifact versions and
hashes remain bound to the original frozen experiment. Test rows decoded/scored: **0**.
Opaque full-source checksum I/O and excluded identity join-key membership checks are necessary
integrity/projection operations, not test feature/label analysis. The isolation tests enforce
bounded record decoding, metadata exclusion, training-only mappings and cold-start semantics.

## Limitations and next phase

These are synthetic development findings. Validation reuse, episode/entity dependence,
few scenario observations, engineered learnability and component correlations limit claims.
No production banking performance or calibrated A/B fraud probabilities can be inferred.
Phase 6 should compare classifier-only against confidence-aware P/A/B ablations, explicit
missing-signal policies and constrained calibrated fusion using TRAIN/VALIDATION only.
Start from hard-negative budgets; anomaly recovers only modest incremental coverage at a
substantial added false-positive cost. Freeze all components/weights/thresholds before
creating/evaluating a fresh future holdout under FINAL_HOLDOUT_PLAN.md. Phase 6 is not implemented.
"""
    )
    for name, content in {
        "ANOMALY_DETECTION.md": anomaly,
        "BEHAVIOURAL_DEVIATION.md": behaviour,
        "PHASE_5_COMPLEMENTARITY.md": complement,
    }.items():
        write_artifact(directory / name, content.encode())
