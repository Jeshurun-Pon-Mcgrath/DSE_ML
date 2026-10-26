"IEEE adapter → existing Phase 1 profile → integration studies and local artifacts."

from pathlib import Path

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.models import LocalDatasetProfile
from riskbyte_ml.data.pipeline import profile_loaded, save_reports
from riskbyte_ml.data.statistics import descriptive_statistics
from riskbyte_ml.datasets.ieee_cis.adapter import load_joined
from riskbyte_ml.datasets.ieee_cis.config import ADAPTER_VERSION, ELAPSED, IEEEConfig
from riskbyte_ml.datasets.ieee_cis.models import IntegrationReport
from riskbyte_ml.datasets.ieee_cis.review import (
    cleaning_review,
    leakage_candidates,
    ordering_reviews,
)
from riskbyte_ml.datasets.ieee_cis.study import (
    behavioural_feasibility,
    entity_key_study,
    feature_catalog,
    feature_contract,
)


def integrate(config: IEEEConfig) -> tuple[LocalDatasetProfile, IntegrationReport]:
    joined = load_joined(config)
    dataset = joined.dataset
    profile = profile_loaded(dataset)
    findings = leakage_candidates(dataset, profile)
    extra = [finding for finding in findings if finding not in profile.warnings]
    warnings = [*profile.warnings, *extra]
    quality = profile.quality.model_copy(
        update={
            "warnings": warnings,
            "validation_status": "FAILED"
            if any(warning.severity == "ERROR" for warning in warnings)
            else "PASSED",
        }
    )
    profile = profile.model_copy(
        update={
            "quality": quality,
            "warnings": warnings,
            "configuration_fingerprint": joined.configuration_fingerprint,
            "pipeline_version": profile.pipeline_version + "/" + ADAPTER_VERSION,
        }
    )
    catalog = feature_catalog(dataset, profile)
    studies = entity_key_study(dataset, config)
    recommendations = sorted(
        (study for study in studies if study.statistically_useful),
        key=lambda study: (
            -(study.repeated_history_row_percentage or 0) * (100 - study.missing_percentage),
            len(study.fields),
            study.name,
        ),
    )
    feasibility = behavioural_feasibility(dataset, studies)
    ordering, rank_correlation = ordering_reviews(dataset, profile)
    report = IntegrationReport(
        dataset_name=profile.dataset_name,
        status="joined",
        fingerprint=dataset.fingerprint,
        source_fingerprints=joined.source_fingerprints,
        adapter_version=ADAPTER_VERSION,
        configuration_fingerprint=joined.configuration_fingerprint,
        join_audit=joined.join_audit,
        memory=joined.memory,
        feature_catalog=catalog,
        entity_key_study=studies,
        recommended_behavioural_groups=[study.name for study in recommendations],
        behavioural_feasibility=feasibility,
        leakage_candidates=findings,
        ordering_review=ordering,
        id_elapsed_rank_correlation=rank_correlation,
        cleaning_review=cleaning_review(profile),
        elapsed_seconds=descriptive_statistics(dataset.frame[ELAPSED], dataset.config),
        temporal_semantics=(
            "TransactionDT is elapsed seconds from an undisclosed reference; "
            "no wall-clock dates, hours, weekdays or months are inferred"
        ),
        feature_contract=feature_contract(catalog, feasibility, profile),
    )
    return profile, report


def _markdown(title: str, payload: str) -> bytes:
    # JSON escaped strings are readable and avoid rendering source values as Markdown/HTML.
    return (
        f"# {title}\n\nDescriptive dataset report; no model fitted.\n\n```json\n"
        + payload
        + "\n```\n"
    ).encode("utf-8")


def save_integration(profile, report: IntegrationReport, directory: Path, config: IEEEConfig):
    documents = {
        "ieee_cis_integration.json": report.model_dump_json(indent=2).encode("utf-8"),
        "join_audit.json": report.join_audit.model_dump_json(indent=2).encode("utf-8"),
        "feature_contract.json": report.feature_contract.model_dump_json(indent=2).encode("utf-8"),
        "ENTITY_KEY_STUDY.md": _markdown(
            "Entity key study",
            report.model_dump_json(
                include={"fingerprint", "entity_key_study", "recommended_behavioural_groups"},
                indent=2,
            ),
        ),
        "IEEE_CIS_LEAKAGE_REVIEW.md": _markdown(
            "IEEE-CIS leakage review",
            report.model_dump_json(
                include={
                    "fingerprint",
                    "leakage_candidates",
                    "ordering_review",
                    "id_elapsed_rank_correlation",
                },
                indent=2,
            ),
        ),
        "IEEE_CIS_BEHAVIOURAL_FEASIBILITY.md": _markdown(
            "IEEE-CIS behavioural feasibility",
            report.model_dump_json(include={"fingerprint", "behavioural_feasibility"}, indent=2),
        ),
        "IEEE_CIS_CLEANING_REVIEW.md": _markdown(
            "IEEE-CIS cleaning review", report.cleaning_review.model_dump_json(indent=2)
        ),
    }
    source_paths = {config.transaction_path.resolve(), config.identity_path.resolve()}
    outputs = [*documents, "data_profile.json", "data_quality_report.md"]
    if any((directory / name).resolve() in source_paths for name in outputs):
        raise DatasetError(
            "dataset_source_overwrite_forbidden", "Reports cannot overwrite either IEEE source"
        )
    save_reports(profile, directory)
    for name, content in documents.items():
        write_artifact(directory / name, content)
