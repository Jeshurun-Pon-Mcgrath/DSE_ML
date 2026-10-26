"""Explicit offline execution; HTTP requests never fit preprocessing or run analysis."""

import json
from datetime import UTC, datetime
from pathlib import Path

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.features.behaviour import build_features
from riskbyte_ml.statistics import ARTIFACT_VERSION, FROZEN_FEATURE_VERSION
from riskbyte_ml.statistics.analysis import (
    correlations,
    describe,
    groups,
    hard_negative_analysis,
    hypotheses,
    prevalence_analysis,
    scenario_analysis,
    temporal_analysis,
)
from riskbyte_ml.statistics.models import Provenance, StatisticalBundle
from riskbyte_ml.statistics.verification import load_verified, verify_sources

LIMITATIONS = [
    "All results are Synthetic-data findings; cannot demonstrate production banking performance.",
    "Generator rules influence observed associations; deterministic "
    "scenarios can make later performance optimistic.",
    "Repeated fictional entities and attack episodes violate "
    "independent-row inference; p-values/FDR and Wilson intervals are "
    "nominal teaching approximations.",
    "BH assumes independent or suitably positively dependent valid "
    "p-values; neither condition is established for this clustered "
    "simulation. Correction does not repair dependence.",
    "No cluster-robust p-values or entity/block bootstrap are implemented; "
    "use those before confirmatory population claims.",
    "Finite-only complete-case analyses report exclusions; device "
    "missingness/cold start changes denominators and can bias comparisons.",
    "Mann-Whitney tests rank/distribution differences, not practical "
    "equivalence or necessarily a difference in medians.",
    "No meaningful effect threshold was prespecified; conventional "
    "magnitude labels are heuristics, not operational utility.",
    "Overall fraud count was fixed by the generator, not sampled IID; "
    "full-dataset prevalence is exact for these records.",
    "Aggregated validation/test data are inspected for authorized "
    "research, so they are no longer unseen for exploratory purposes. Do "
    "not tune later model/threshold choices to these summaries.",
    "Synthetic entities are not bank customers; concept drift is "
    "approximated. Future ABRF results can describe this simulation only.",
    "Fraud-scenario and hard-negative fields remain evaluation-only; no "
    "labels enter prior-history calculations or predictive features.",
]


def generate(config, artifact_root: Path, output: Path) -> StatisticalBundle:
    frame, evaluation, split, manifest = load_verified(config, artifact_root)
    sources = {
        config.transaction_path.resolve(),
        config.identity_path.resolve(),
        config.metadata_path.resolve(),
        (config.transaction_path.parent / "evaluation_metadata.csv").resolve(),
    }
    if output.resolve() == config.transaction_path.parent.resolve() or any(
        output.resolve() in p.parents for p in sources
    ):
        raise DatasetError(
            "statistics_output_invalid", "Statistical outputs must be separate from dataset sources"
        )
    features = build_features(frame, horizon_seconds=manifest["config"]["duration_days"] * 86400)
    rows = split.indexes["train"]
    bundle = StatisticalBundle(
        provenance=Provenance(
            dataset_fingerprint=manifest["fingerprint"],
            feature_pipeline_version=FROZEN_FEATURE_VERSION,
            split_id=split.split_id,
            artifact_version=ARTIFACT_VERSION,
            generated_at=datetime.now(UTC),
            limitations=LIMITATIONS,
        ),
        descriptive=describe(features, groups(frame.isFraud, evaluation)),
        hypotheses=hypotheses(
            features.iloc[rows].reset_index(drop=True),
            frame.isFraud.iloc[rows].reset_index(drop=True),
            evaluation.iloc[rows].reset_index(drop=True),
        ),
        correlations=correlations(features.iloc[rows]),
        prevalence=prevalence_analysis(frame.isFraud, split.indexes),
        scenarios=scenario_analysis(features, frame.isFraud, evaluation),
        hard_negatives=hard_negative_analysis(features, frame.isFraud, evaluation),
        temporal=temporal_analysis(features, frame, evaluation, split),
    )
    verify_sources(config)
    files = {
        "descriptive_statistics": bundle.descriptive,
        "hypothesis_tests": bundle.hypotheses,
        "correlations": bundle.correlations,
        "prevalence": bundle.prevalence,
        "hard_negative_analysis": bundle.hard_negatives,
        "scenario_analysis": bundle.scenarios,
        "temporal_analysis": bundle.temporal,
    }
    for name, report in files.items():
        write_artifact(
            output / (name + ".json"),
            json.dumps(
                {
                    "provenance": bundle.provenance.model_dump(mode="json"),
                    "data": report.model_dump(mode="json"),
                },
                indent=2,
                allow_nan=False,
            ).encode(),
        )
    write_artifact(output / "STATISTICAL_ANALYSIS.md", render_report(bundle).encode())
    # Single atomic publication point: API reads this coherent bundle, never mixed side reports.
    write_artifact(output / "analysis_bundle.json", bundle.model_dump_json(indent=2).encode())
    return bundle


def number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def render_report(bundle):
    lines = [
        "# RiskByte statistical intelligence — Synthetic-data findings",
        "",
        "All measurements below describe the IEEE-CIS-inspired synthetic "
        "financial transaction dataset. "
        "They are not IEEE-CIS observations, banking-population estimates or "
        "model performance.",
        "",
        f"Dataset fingerprint: `{bundle.provenance.dataset_fingerprint}`.",
        f"Feature version: `{bundle.provenance.feature_pipeline_version}`.",
        f"Frozen temporal split: `{bundle.provenance.split_id}`.",
        f"Statistical artifact version: `{bundle.provenance.artifact_version}`.",
        f"Generated UTC: `{bundle.provenance.generated_at.isoformat()}`.",
        "",
        "## Execution and statistical design",
        "",
        "Read-only source verification hashes both CSVs plus generator "
        "configuration/version and the independently "
        "hashed evaluation sidecar. Frozen feature/split IDs, split hash, "
        "disjoint/exhaustive indexes, chronological "
        "boundaries, row-order hash and exact target/split counts are verified "
        "before analysis. A mismatch aborts "
        "execution without publishing new results. No models, imputers, "
        "encoders, SHAP or ABRF are fitted.",
        "",
        "Primary six hypotheses use only the frozen training partition. "
        "Descriptives, scenario/hard-negative "
        "exploration and elapsed-time summaries cover all records. "
        "Correlations use training observations. "
        "Holdout aggregates are exploratory and cannot be used for selecting "
        "later models or thresholds.",
        "",
        "Sample variance/std use ddof=1; history features retain Phase 2 "
        "population std semantics. Percentiles "
        "use linear interpolation. Skewness and Fisher excess kurtosis are "
        "bias adjusted where defined; "
        "constant/small/all-missing groups return null for undefined moments. "
        "Missing/invalid/infinite values "
        "are excluded with counts, never imputed for tests. Zero means/std in "
        "history remain unavailable. "
        "Group quartiles support box plots; no individual raw records or huge "
        "matrices enter APIs.",
        "",
        "Two-sided Mann-Whitney U handles unequal skewed/discrete groups; "
        "asymptotic tests correct ties and "
        "continuity. Small untied samples use exact tests, small tied samples "
        "seeded permutations. Group "
        "shapes are reported instead of claiming normality or equal variances. "
        "This is not a median-only test "
        "when shapes differ. Practical equivalence in the requested RQ1 "
        "wording is not testable without a "
        "prespecified equivalence margin; the operational H0 is equality of distributions.",
        "",
        "Binary novelty tests exclude unavailable context, use Pearson "
        "chi-square with expected-count checks "
        "and Fisher exact for sparse 2x2 tables. Cliff's delta is P(fraud > "
        "reference)-P(fraud < reference), "
        "with ties zero. Cramer's V is unsigned; novelty odds ratios compare "
        "fraud odds for new versus seen "
        "context. Zero-denominator odds are null with reason. No arbitrary "
        "pseudocount is added.",
        "",
        "Alpha=0.05. Benjamini-Hochberg FDR is applied once across all six "
        "prespecified questions (unavailable "
        "questions remain in the family with p=1 for adjustment). Report "
        "raw/adjusted p-values and effects. "
        "Repeated entities/episodes violate IID assumptions: all decisions "
        "below are nominal exploratory "
        "decisions. FDR cannot repair invalid independence assumptions; "
        "significance is not business utility.",
        "A computed p-value of 0 denotes floating-point underflow, not a mathematical zero. "
        "The machine-readable p_value_underflow flag records this explicitly.",
        "",
        "## Research questions and results",
        "",
        "| Question | Feature | Test | Raw p | BH adjusted p | Effect | Nominal decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in bundle.hypotheses.hypotheses:
        effect = h.outcome.effect_size
        lines.append(
            f"| {h.id} | {h.feature} | {h.outcome.method} | {number(h.outcome.p_value)} | "
            f"{number(h.adjusted_p_value)} | {effect.method if effect else 'unavailable'} "
            f"{number(effect.value) if effect else ''} | {h.decision} |"
        )
    for h in bundle.hypotheses.hypotheses:
        lines.extend(
            [
                "",
                f"### {h.id}: {h.question}",
                "",
                f"H0: {h.h0}.",
                "",
                f"H1: {h.h1}.",
                "",
                f"Selection: {h.test_selection_reasoning}.",
                "",
                h.practical_interpretation,
                "",
                f"Statistic: {number(h.outcome.statistic)}; "
                f"group counts: `{h.outcome.group_counts}`; "
                f"excluded: `{h.outcome.excluded_counts}`.",
                "",
                "Assumption checks: " + " ".join(h.outcome.assumption_checks),
            ]
        )
        if h.outcome.effect_size and h.outcome.effect_size.method == "Cramers_V":
            lines.extend(
                [
                    "",
                    f"Odds ratio: {number(h.outcome.effect_size.odds_ratio)}. "
                    f"Table: `{h.outcome.contingency_table}`; {h.outcome.table_layout}.",
                ]
            )
        for label, s in h.outcome.group_summaries.items():
            lines.extend(
                [
                    "",
                    f"{label}: mean {number(s.mean)}, median {number(s.median)}, "
                    f"variance {number(s.variance)}, std {number(s.standard_deviation)}. "
                    f"{s.distribution_note}.",
                ]
            )
    lines.extend(
        [
            "",
            "## Descriptive group comparison — Synthetic-data findings",
            "",
            "Full-group statistics include all requested raw and history "
            "variables, quartiles, percentiles, "
            "variance, standard deviation, skewness and kurtosis in "
            "descriptive_statistics.json.",
            "",
            "| Group | Rows | Amount mean | Amount median | Amount variance | Amount std |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for group, values in bundle.descriptive.groups.items():
        s = values["TransactionAmt"]
        lines.append(
            f"| {group} | {bundle.descriptive.group_counts[group]} | {number(s.mean)} | "
            f"{number(s.median)} | {number(s.variance)} | {number(s.standard_deviation)} |"
        )
    lines.extend(
        [
            "",
            "## Fraud prevalence — Synthetic-data findings",
            "",
            "| Partition | Rows | Fraud | Prevalence | Nominal 95% Wilson CI |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, p in bundle.prevalence.partitions.items():
        lines.append(
            f"| {name} | {p.rows} | {p.fraud_count} | {100 * p.prevalence:.4f}% | "
            f"{100 * p.lower:.4f}%–{100 * p.upper:.4f}% |"
        )
    lines.extend(
        [
            "",
            bundle.prevalence.interpretation,
            "",
            "## Hard negatives — Synthetic-data findings",
            "",
            "| Signal | Normal legitimate median | Hard-negative median | Fraud "
            "median | Fraud vs hard-negative delta |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, effect in bundle.hard_negatives.fraud_vs_hard_negative_effects.items():
        g = bundle.hard_negatives.groups
        lines.append(
            f"| {name} | {number(g['normal_legitimate'].numeric[name].median)} | "
            f"{number(g['hard_negative_legitimate'].numeric[name].median)} | "
            f"{number(g['fraudulent'].numeric[name].median)} | {number(effect.value)} |"
        )
    lines.extend(
        [
            "",
            "| Group | Device novelty fraction (available rows) | Merchant novelty "
            "| Region novelty |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, g in bundle.hard_negatives.groups.items():
        lines.append(
            f"| {name} | {number(g.novelty['device_seen_before'].new_fraction)} | "
            f"{number(g.novelty['merchant_seen_before'].new_fraction)} | "
            f"{number(g.novelty['billing_region_seen_before'].new_fraction)} |"
        )
    lines.extend(
        [
            "",
            "Synthetic-data finding: aggregate hard negatives overlap fraud in amount "
            "deviation. The training RQ6 delta is negligible despite a small nominal p-value. "
            "Device/region novelty and failed-attempt distributions show group differences, "
            "but no signal alone establishes fraud or classifier separability.",
            "",
            "| Hard-negative subtype | Amount median | Prior-mean amount ratio median | "
            "30-minute velocity median |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, g in bundle.hard_negatives.hard_negative_types.items():
        lines.append(
            f"| {name} | {number(g.numeric['TransactionAmt'].median)} | "
            f"{number(g.numeric['amount_ratio_to_prior_mean'].median)} | "
            f"{number(g.numeric['transactions_prior_30m'].median)} |"
        )
    lines.extend(
        [
            "",
            "Synthetic-data finding: laptop and medical hard negatives have high amount "
            "ratios; shopping-festival hard negatives have high velocity. Legitimate new-device "
            "and travel events share novelty signals with attacks. These subtype summaries "
            "demonstrate why future evaluation must measure hard-negative false positives.",
        ]
    )
    lines.extend(
        [
            "",
            "Subtype-level amount/velocity/deviation/novelty summaries in "
            "hard_negative_analysis.json "
            "identify which unusual legitimate mechanisms resemble fraud. Similar "
            "medians or small deltas "
            "indicate overlap, not classifier separability; distribution tails and "
            "missing evidence matter.",
            "",
            "## Scenarios — Synthetic-data findings",
            "",
            "| Scenario | Count | Fraud share | Amount median | 30-minute velocity median |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, g in bundle.scenarios.scenarios.items():
        lines.append(
            f"| {name} | {g.count} | {g.proportion:.4f} | "
            f"{number(g.numeric['TransactionAmt'].median)} | "
            f"{number(g.numeric['transactions_prior_30m'].median)} |"
        )
    lines.extend(
        [
            "",
            "Scenario labels describe synthetic generating mechanisms only; they "
            "never enter feature inputs. "
            "Episode sizes explain unequal row shares. Novelty summaries retain "
            "available/unavailable denominators.",
            "",
            "## Correlations — Synthetic-data findings",
            "",
            "Pearson and Spearman are reported for 66 fixed meaningful training "
            "pairs with finite paired counts. "
            "Spearman is preferred for skewed monotonic behaviour; neither implies "
            "causation. No identifiers, "
            "target or evaluation metadata enter the matrix. Strongest absolute Spearman pairs:",
            "",
        ]
    )
    ranked = sorted(bundle.correlations.pairs, key=lambda p: abs(p.spearman or 0), reverse=True)[:5]
    lines.extend(
        f"- {p.first} / {p.second}: Spearman {number(p.spearman)}, "
        f"Pearson {number(p.pearson)}, n={p.pair_count}."
        for p in ranked
    )
    temporal = bundle.temporal.partition_fraud_association
    lines.extend(
        [
            "",
            "## Temporal exploration — Synthetic-data findings",
            "",
            "Elapsed-day volume, prevalence, scenario counts and partition feature "
            "summaries are saved in "
            "temporal_analysis.json. Only relative synthetic time is used; no "
            "dates/weekdays/months.",
            "",
            f"Supplementary partition prevalence association: {temporal.method}, statistic "
            f"{number(temporal.statistic)}, raw p={number(temporal.p_value)}; "
            "Cramer's V="
            f"{number(temporal.effect_size.value) if temporal.effect_size else 'unavailable'}. "
            "This exploratory diagnostic is outside the six-question FDR family "
            "and is not a confirmatory claim.",
            "",
            "## Reproduction, API and artifacts",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m riskbyte_ml.statistics --config "
            "ml/config/synthetic_riskbyte.example.json",
            "```",
            "",
            "Default artifact root is ml/artifacts; output is "
            "ml/artifacts/statistics. Individual JSON files "
            "carry identical provenance; analysis_bundle.json is published last "
            "atomically as the coherent API "
            "snapshot. One writer per output directory. API rechecks exact source "
            "hashes and frozen provenance; "
            "no analysis runs in a request, no local paths or raw rows are "
            "returned. Missing configuration is "
            "dataset_not_configured; a verified source without a bundle is "
            "statistics_not_generated.",
            "",
            "GET /api/v1/statistics/summary, /hypotheses, /correlations, "
            "/prevalence, /scenarios, "
            "/hard-negatives and /temporal provide typed finite responses. "
            "Quartiles, effect/statistic/p-value "
            "fields, pairwise coefficients, group/scenario summaries and "
            "elapsed-day buckets support later "
            "visualizations. Undefined statistics remain null with counts/reasons; "
            "no frontend redesign.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend("- " + item for item in bundle.provenance.limitations)
    lines.extend(
        [
            "",
            "## References",
            "",
            "Test implementations follow [SciPy Mann-Whitney "
            "U](https://docs.scipy.org/doc/scipy/reference/generated/"
            "scipy.stats.mannwhitneyu.html), "
            "[chi-square](https://docs.scipy.org/doc/scipy/reference/generated/"
            "scipy.stats.contingency.chi2_contingency.html) "
            "and [Fisher "
            "exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).",
            "",
            "## Exact next recommended phase",
            "",
            "Review these frozen exploratory findings, then separately authorize "
            "Phase 4 supervised baseline "
            "classification using the existing temporal split and train-only "
            "preprocessing. Report scenario recall "
            "and hard-negative false-positive rates; freeze model decisions "
            "without tuning against inspected test "
            "summaries. No Phase 4 work is performed here.",
            "",
        ]
    )
    return "\n".join(lines)
