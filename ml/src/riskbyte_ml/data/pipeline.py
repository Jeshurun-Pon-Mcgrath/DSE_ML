"""Reproducible inspection → validation → statistics → EDA → reviewable reports."""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.cleaning import propose_cleaning
from riskbyte_ml.data.config import DatasetConfig
from riskbyte_ml.data.eda import generate_eda
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import LoadedDataset, load_csv
from riskbyte_ml.data.inspection import analytical_numeric_columns, inspect_schema
from riskbyte_ml.data.models import LocalDatasetProfile
from riskbyte_ml.data.statistics import descriptive_statistics
from riskbyte_ml.data.validation import validate_dataset

PIPELINE_VERSION = "data-intelligence-v1"


def profile_dataset(config: DatasetConfig) -> LocalDatasetProfile:
    return profile_loaded(load_csv(config))


def profile_loaded(dataset: LoadedDataset) -> LocalDatasetProfile:
    """Reuse Phase 1 reporting for validated adapter frames without reloading/copying."""
    config = dataset.config
    schema = inspect_schema(dataset)
    quality = validate_dataset(dataset, schema)
    statistics = {
        name: descriptive_statistics(dataset.frame[name], config)
        for name in analytical_numeric_columns(schema, config)
    }
    config_bytes = json.dumps(
        config.model_dump(mode="json", exclude={"source_path"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return LocalDatasetProfile(
        dataset_name=config.name,
        source_path=str(config.source_path.resolve()),
        fingerprint=dataset.fingerprint,
        configuration_fingerprint="sha256:" + hashlib.sha256(config_bytes).hexdigest(),
        generated_at=datetime.now(UTC),
        pipeline_version=PIPELINE_VERSION,
        library_versions={
            name: version(name) for name in ("riskbyte-ml", "pandas", "numpy", "pydantic")
        },
        row_count=len(dataset.frame),
        column_count=len(dataset.frame.columns),
        memory_usage_bytes=int(dataset.frame.memory_usage(index=True, deep=True).sum()),
        schema_metadata=schema,
        descriptive_statistics=statistics,
        quality=quality,
        unique_counts={column.name: column.unique_count for column in schema.columns},
        data_type_summary={
            "numeric": schema.numeric_columns,
            "categorical": schema.categorical_columns,
            "boolean": schema.boolean_columns,
            "datetime_like": schema.datetime_like_columns,
        },
        warnings=quality.warnings,
        cleaning_plan=propose_cleaning(dataset, quality),
        eda=generate_eda(dataset, schema, quality, statistics),
    )


def quality_markdown(profile: LocalDatasetProfile) -> str:
    def escape(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "\\|")
            .replace("\n", " ")
            .replace("\r", " ")
        )

    lines = [
        "# RiskByte data quality report",
        "",
        f"Dataset: {escape(profile.dataset_name)}",
        f"Fingerprint: `{profile.fingerprint}`",
        f"Configuration: `{profile.configuration_fingerprint}`",
        f"Generated: {profile.generated_at.isoformat()}",
        f"Pipeline: {profile.pipeline_version}",
        f"Rows: {profile.row_count}; columns: {profile.column_count}",
        f"Validation: **{profile.quality.validation_status}**",
        "",
        "Source values are preserved. Heuristics are candidates; nothing was "
        "cleaned automatically.",
        "",
        "| Severity | Code | Column | Affected rows | Inferred | Explanation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for warning in profile.warnings:
        cells = [
            warning.severity,
            warning.code,
            warning.column or "—",
            warning.rows_affected if warning.rows_affected is not None else "—",
            warning.inferred,
            warning.explanation,
        ]
        lines.append("| " + " | ".join(escape(value) for value in cells) + " |")
    if not profile.warnings:
        lines.append(
            "| INFO | no_findings | — | — | False | No configured checks reported findings |"
        )
    lines.extend(["", "## EDA availability", ""])
    for name in type(profile.eda).model_fields:
        analysis = getattr(profile.eda, name)
        lines.append(
            f"- {name}: {analysis.status}" + (f" ({analysis.reason})" if analysis.reason else "")
        )
    lines.extend(
        [
            "",
            "## Cleaning philosophy",
            "",
            "Review the source-bound cleaning plan in data_profile.json. Duplicate removal may",
            "remove legitimate repeated events. Outliers are reported only; they "
            "may be fraud signals.",
            "",
            "Percentages use valid configured labels. Undefined ratios/moments are "
            "null, never Infinity.",
            "",
        ]
    )
    return "\n".join(lines)


def save_reports(profile: LocalDatasetProfile, directory: Path) -> None:
    paths = [directory / "data_profile.json", directory / "data_quality_report.md"]
    if any(path.resolve() == Path(profile.source_path).resolve() for path in paths):
        raise DatasetError(
            "dataset_source_overwrite_forbidden", "Report paths cannot overwrite source"
        )
    write_artifact(paths[0], profile.model_dump_json(indent=2).encode("utf-8"))
    write_artifact(paths[1], quality_markdown(profile).encode("utf-8"))
