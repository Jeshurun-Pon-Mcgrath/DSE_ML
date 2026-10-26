"""Report-ready tables from actual frozen-score validation, never the original test."""

import json

from riskbyte_ml.data.artifacts import write_artifact


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def pct(value):
    return f"{value:.4%}" if value is not None else "unavailable"


def metric_rows(bundle):
    rows = []
    for entry in bundle.ablation:
        metric = entry["metrics"]
        label = entry["variant"]
        if label == "Supervised":
            label += " (policy-optimized control)"
        if metric is None:
            rows.append([label] + ["unavailable"] * 7)
            continue
        rows.append(
            [
                label,
                pct(metric["precision"]),
                pct(metric["recall"]),
                f"{metric['f1']:.6f}",
                f"{metric['pr_auc']:.6f}" if metric["pr_auc"] is not None else "N/A: binary",
                pct(metric["fpr"]),
                pct(metric["hard_negatives"]["rate"]),
                metric["threshold"],
            ]
        )
    return rows


def write_reports(bundle, directory):
    s, m, i = bundle.selected, bundle.validation, bundle.improvement
    ablation = table(
        ["Variant", "Precision", "Recall", "F1", "AP", "FPR", "Hard-negative FPR", "Threshold"],
        metric_rows(bundle),
    )
    scenario = table(
        [
            "Scenario",
            "N",
            "Frozen supervised recall",
            "Frozen anomaly recall",
            "Selected ABRF recall",
        ],
        [
            [
                name,
                row["count"],
                pct(row["supervised"]["rate"]),
                pct(row["anomaly"]["rate"]),
                pct(row["abrf"]["rate"]),
            ]
            for name, row in bundle.scenario_analysis.items()
        ],
    )
    hard = table(
        [
            "Hard-negative type",
            "N",
            "Frozen P flags",
            "ABRF flags",
            "ABRF FPR",
            "P/A/B crossing allocations",
        ],
        [
            [
                name,
                row["abrf"]["count"],
                row["supervised"]["flagged"],
                row["abrf"]["flagged"],
                pct(row["abrf"]["rate"]),
                row["crossing_allocations"],
            ]
            for name, row in bundle.hard_negative_analysis.items()
        ],
    )

    def study_rows(study):
        rows = []
        for name, row in study.items():
            if row.get("status") != "AVAILABLE":
                rows.append([name, "unavailable", "", "", "", ""])
                continue
            metric = row["metrics"]
            rows.append(
                [
                    name,
                    row["weights"],
                    row["curve"],
                    f"{metric['f1']:.6f}",
                    f"{metric['pr_auc']:.6f}",
                    pct(metric["hard_negatives"]["rate"]),
                ]
            )
        return rows

    study_headers = ["Study", "Weights", "Confidence", "F1", "AP", "Hard-negative FPR"]
    constraint = table(study_headers, study_rows(bundle.constraint_study))
    confidence = table(study_headers, study_rows(bundle.confidence_study))
    policies = table(study_headers, study_rows(bundle.policies))
    provenance = f"""**Synthetic-data findings only.**
Dataset `{bundle.provenance["dataset_fingerprint"]}`;
feature pipeline `{bundle.provenance["feature_version"]}`; split `{bundle.provenance["split_id"]}`.
Version `{bundle.version}`. Only the existing 70,000 TRAIN / 15,000 VALIDATION score artifacts
are decoded. The original temporal test is not used or scored. All component bytes,
normalization snapshots, versions and thresholds are verified before and after research.
No Decision Tree, Isolation Forest or behavioural component is fitted or modified.

"""
    research = (
        "# ABRF research\n\n"
        + provenance
        + f"""## Motivation and exact formula

Adaptive Behavioural Risk Fusion is a RiskByte-specific, transparent fusion methodology,
not a claim of universally optimal or novel mathematics. P is the frozen calibrated
supervised probability; A is global anomaly and B entity-relative deviation. A/B/R are NOT
fraud probabilities. Phase 4's frozen tree is precise but misses some velocity/weak-signal
patterns. Phase 5's analysis recovered 14 additional validation fraud rows at 68 added false
positives. High deviation alone cannot justify a fraud decision.

Let `g(C)` be the selected confidence transform and `I_B` behaviour availability.
`d = wp + wa + wb * I_B * g(C_b)`.
`R = (wp*P + wa*A + wb*I_B*g(C_b)*B) / d`.
For unavailable B the behavioural term is omitted, not evaluated with an assumed safe B=0.
Effective weights are `(wp/d, wa/d, wb*I_B*g(C_b)/d)`; contributions multiply those weights
by the respective available signals. Their sum is R. Weights are finite/nonnegative,
sum to 1 and retain positive P/A fallback mass. P/A are required; no undocumented missing-P/A
fallback exists. If C_b=0, B has no influence even when a score exists. C_b=1 gives full base
behavioural weight. Dynamic weighting responds to reliability, not transaction outcomes.

## Confidence and bounded search

Compare linear `g(C)=C`, and gates `g(C)=C if C>=0.5 else 0` / `C>=0.75`.
There is no neural/meta-classifier or behavioural-component retuning. Confidence is the
unchanged Phase 5 count/availability measure, not calibrated confidence in a fraud claim.
All grid candidates and confidence rules are specified before inspecting Phase 6 results.
Grid divisions: {bundle.search_configuration.grid_divisions}; each P weight is positive.
Zero A/B weights are eligible so ablations can reject a component whose value is unsupported.
Evaluated **{bundle.search_summary["unique_weight_combinations"]}** weight combinations,
**{bundle.search_summary["candidate_configurations"]}** weight/curve candidates,
**{bundle.search_summary["thresholds_evaluated"]:,}** exact threshold locations and
**{bundle.search_summary["policy_configurations"]}** candidate-policy configurations.
All threshold metrics, including AP/FPR/FNR/hard-negative FPR/TP/FP/FN, are stored in
compressed per-candidate CSV files rather than huge API payloads. AP is constant across
thresholds for a given ranking and means non-interpolated average precision.

## Operating policies, fairness and selection

BALANCED maximizes F1 under FPR≤1%, hard-negative FPR≤5%.
HIGH_RECALL maximizes recall under FPR≤2%, hard-negative FPR≤10%.
STRICT minimizes false positives with recall≥60%, FPR≤0.5%, hard-negative FPR≤3%.
Phase 6 deliberately tightens BALANCED hard-negative control versus earlier 10% conventions.
Every variant uses the same validation rows, budgets, confidence choices where applicable
and exact threshold search. Protected scenario floors require at least the frozen P detections
for account takeover/mixed mode, identically across variants. Other regressions are reported.
The default policy **{s.policy}** is declared before search, to favor precision/recall balance
and stronger hard-negative control. Candidate ranking uses its policy objective, then AP,
precision, recall, subgroup/overall FPR and simpler confidence logic; accuracy is not primary.
Scenario metadata controls offline evaluation/protection only and never enters live fusion.

The P-only policy threshold is a Phase 6 control, NOT a change to Phase 4's frozen threshold.
Both are reported. Frozen binary OR uses the unchanged P/A thresholds and has no meaningful
continuous ranking AP; its AP is deliberately unavailable rather than relabelled.

{ablation}

## Supervised-priority and confidence comparisons

The full three-signal candidates are compared unrestricted, with `wp>=wa,wp>=wb`, and
with `wp>=0.4`, using the same policy. These filters do not receive separate special tuning.
No constraint is claimed to improve robustness unless observed measurements support it.

{constraint}

{confidence}

Other policy operating points:

{policies}

## Frozen final development configuration

```json
{json.dumps(s.model_dump(mode="json"), indent=2)}
```

{bundle.selection_rationale}

Selected precision {pct(m["precision"])}, recall {pct(m["recall"])}, F1 {m["f1"]:.6f},
AP {m["pr_auc"]:.6f}, ROC-AUC {m["roc_auc"]:.6f}, FPR {pct(m["fpr"])},
hard-negative FPR {pct(m["hard_negatives"]["rate"])}. Confusion: {m["confusion"]}.
Compared with frozen P: {i["additional_fraud_recovered"]} newly recovered fraud rows,
{i["previously_detected_fraud_lost"]} previously detected fraud lost,
net fraud change {i["net_fraud_detection_change"]}; {i["additional_false_positives"]} added
false positives, {i["previous_false_positives_removed"]} previous false positives removed,
net FP change {i["net_false_positive_change"]}, net hard-negative FP change
{i["net_hard_negative_false_positive_change"]}.

## Risk bands, trace and calibration

Risk boundaries are TRAIN fused-score quantiles {bundle.search_configuration.risk_band_quantiles},
without label use: **{s.risk_boundaries}**. LOW/GUARDED/ELEVATED/HIGH/CRITICAL bins are
left-inclusive at internal boundaries; R=1 is CRITICAL. The investigation threshold is a
separate validation operating choice; a risk category is not a statement of guilt.
Quantile risk bands are development categories, not validated banking risk bands.

`FusionInput` excludes all target/scenario/hard-negative fields. `fuse` exposes validated
inputs, base/effective weights, contributions, R, category, threshold, flag and deterministic
evidence codes. Optional novelty/velocity/amount reason codes require supplied available
structured component evidence. The existing score sidecar lacks these fine-grained signals,
so research traces do not invent them. Contributions and subtractive crossing allocations
are calculation explanations, not SHAP or causal attribution. Multiple components can be
necessary for a crossing; counts need not be mutually exclusive.

No calibration is applied to R. A strictly monotonic calibration preserves ranking and
available threshold decisions, so it cannot improve AP/recall-FPR tradeoffs by itself.
Isotonic ties can change ranking and require a separately justified calibration protocol.
No empirical Brier improvement is claimed. R remains a normalized decision-support score;
future probability claims would require independent calibration evidence.

## Behaviour-only support and scenario/hard-negative findings

Frozen P-negative/A-negative/B-high support: {bundle.support_analysis}.
Signals overlap unusual legitimate transactions, so high B does not automatically imply fraud.

{scenario}

{hard}

Improved scenarios: {i["scenarios_improved"]}. Degraded scenarios: {i["scenarios_degraded"]}.
International separability is described using saved validation P/A/B/confidence distributions;
no international-fraud rule is added. Low-and-slow and velocity results must be read alongside
global/hard-negative costs, not selected as isolated success measures.

## Reproducibility, API, final holdout and limitations

Ignored `ml/artifacts/abrf/<run-id>/` stores selected configuration, provenance/selection
metadata, all ranked candidates, threshold CSVs, development scores, a computed trace and
coherent report bundle. Exclusive writes preserve history; changed searches cannot silently
replace the active selected configuration. Cached runs verify all checksums without refitting.
GET `/api/v1/abrf/{{status,config,validation,ablation,scenarios,hard-negatives}}` serves
read-only typed JSON reports, never a public prediction API or raw transaction records.

Fresh holdout status: **{bundle.future_holdout_status}**. Generation was explicitly deferred;
current generator lacks reviewed stateful future continuation. FINAL_HOLDOUT_PLAN.md remains
the governing protocol. No fresh dataset is generated or scored in this phase.

Synthetic data cannot prove production banking performance. Generator rules affect learnability;
confidence is provisional; P/A/B share correlated signals; validation was already reused in
earlier phases and is reused for this search, so results are development findings, not unbiased
generalization. Scenario counts are small and episodes/entities dependent. Overall subgroup
caps do not guarantee every subtype's rate. Missing-history fallback and shift in confidence
or score percentiles need fresh future stress testing. ABRF is project-specific, not universally
optimal. No SHAP, final inference endpoint, UI redesign or Phase 7 implementation occurs.
"""
    )
    validation = (
        "# ABRF validation tables\n\n"
        + provenance
        + f"""## Component ablations

{ablation}

P-only optimization is a fair control; the historical Phase 4 threshold remains unchanged.
Naive OR ranking AP is unavailable because it emits only a binary union, not an R score.

## Scenario recall

{scenario}

## Hard-negative results and contribution crossings

{hard}

## Improvement accounting

```json
{json.dumps(i, indent=2)}
```

## Selection and reproducibility

Configuration: `{bundle.selected_candidate_id}`, version `{bundle.version}`, policy `{s.policy}`.
Base weights `{s.weights.model_dump()}`, confidence `{s.confidence_curve}`, threshold
`{s.decision_threshold}`. Risk boundaries `{s.risk_boundaries}`.
{bundle.selection_rationale}

```powershell
.\\.venv\\Scripts\\python.exe -m riskbyte_ml.abrf `
  --config ml/config/synthetic_riskbyte.example.json `
  --search-config ml/config/abrf.example.json
```

Original temporal test accessed for ABRF fitting/tuning/scoring: **NO**.
Opaque full-source hashing verifies provenance without decoding original transaction records.
Fresh final holdout: **not generated or evaluated**. Models and component thresholds unchanged.
All figures are synthetic validation development observations; validation reuse, generator
bias, dependence and incomplete context prevent production banking claims.
"""
    )
    for name, content in {"ABRF_RESEARCH.md": research, "ABRF_VALIDATION.md": validation}.items():
        write_artifact(directory / name, content.encode())
