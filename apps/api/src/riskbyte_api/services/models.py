"""Read-only strict evaluation registry; never train, score transactions or unpickle."""

from riskbyte_api.core.config import REPOSITORY_ROOT
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.models.supervised import MODEL_VERSION, REGISTRY_VERSION
from riskbyte_ml.models.supervised.artifacts import verify_registry_artifacts
from riskbyte_ml.models.supervised.inputs import verify_frozen_files
from riskbyte_ml.models.supervised.pipeline import ranking
from riskbyte_ml.models.supervised.schemas import (
    ModelComparison,
    ModelComparisonRow,
    ModelMetricView,
    ModelOverview,
    ModelRegistry,
    ModelResponse,
    ScenarioRecallView,
)
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import verify_sources


class ModelService:
    def __init__(self, settings):
        self.settings = settings

    @staticmethod
    def resolve(path):
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    def response(self, section, model_id=None):
        if self.settings.dataset_config_path is None:
            return ModelResponse(status="dataset_not_configured")
        if self.settings.dataset_adapter != "synthetic_riskbyte":
            return ModelResponse(
                status="models_unavailable", reason="Frozen synthetic source required"
            )
        verify_sources(load_synthetic_config(self.resolve(self.settings.dataset_config_path)))
        directory = self.resolve(self.settings.model_artifacts_dir)
        verify_frozen_files(directory.parent)
        path = directory / "model_registry.json"
        if not path.is_file():
            return ModelResponse(status="models_not_trained")
        try:
            if path.stat().st_size > 15_000_000:
                raise ValueError
            registry = ModelRegistry.model_validate_json(path.read_bytes())
            p = registry.provenance
            if (
                p.dataset_fingerprint != FROZEN_FINGERPRINT
                or p.feature_version != FROZEN_FEATURE_VERSION
                or p.split_id != FROZEN_SPLIT_ID
                or p.model_version != MODEL_VERSION
                or p.registry_version != REGISTRY_VERSION
            ):
                raise ValueError
        except (OSError, ValueError):
            raise DatasetError(
                "model_registry_invalid", "Model evaluation registry is invalid or incompatible"
            ) from None
        verify_registry_artifacts(registry, directory)
        selected = next(m for m in registry.models if m.is_selected_supervised_model)
        if section == "list":
            data = [self.overview(m) for m in registry.models]
        elif section == "comparison":
            data = ModelComparison(
                rows=[
                    ModelComparisonRow(
                        **self.overview(m).model_dump(),
                        selection_metrics=m.selection.metrics if m.selection else None,
                        validation_metrics=m.validation.metrics if m.validation else None,
                        hard_negative_performance=m.validation.hard_negatives
                        if m.validation
                        else None,
                        scenario_recall=m.validation.scenario_recall if m.validation else {},
                        timing=m.timing,
                        class_imbalance_strategy=m.class_imbalance_strategy,
                    )
                    for m in registry.models
                ],
                pr_auc_ranking=[m.model_id for m in ranking(registry.models)],
                hard_negative_fpr_ranking=[
                    m.model_id
                    for m in sorted(
                        [
                            m
                            for m in registry.models
                            if m.validation and m.validation.hard_negatives.rate is not None
                        ],
                        key=lambda m: m.validation.hard_negatives.rate,
                    )
                ],
                selection_rationale=registry.selection_freeze.rationale,
                note="Synthetic-data findings. Primary ranking uses later-validation "
                "selection block; whole validation includes calibration-fitting "
                "rows. Only selected model scored on primary test.",
            )
        elif section == "selected":
            data = selected
        else:
            matches = [m for m in registry.models if m.model_id == model_id]
            if not matches:
                raise DatasetError("model_not_found", "Model ID is not registered", 404)
            record = matches[0]
            if record.status == "DEPENDENCY_UNAVAILABLE":
                return ModelResponse(status="models_unavailable", reason=record.limitation)
            reason = (
                None
                if record.test_metrics
                else "Primary test reserved for the frozen selected classifier only"
            )
            if section == "metrics":
                data = ModelMetricView(
                    train=record.train_metrics,
                    validation=record.validation,
                    selection=record.selection,
                    final_test=record.test_metrics,
                    calibration=record.calibration,
                    test_availability_reason=reason,
                )
            elif section == "scenarios":
                data = ScenarioRecallView(
                    validation=record.validation.scenario_recall,
                    selection=record.selection.scenario_recall,
                    final_test=record.test_metrics.scenario_recall if record.test_metrics else None,
                    test_availability_reason=reason,
                )
            elif section == "thresholds":
                if record.threshold_analysis is None:
                    return ModelResponse(
                        status="models_unavailable",
                        reason="Fixed rule baseline has no learned threshold search",
                    )
                data = record.threshold_analysis
            else:
                raise ValueError("Unknown model report section")
        return ModelResponse(status="models_ready", data=data, provenance=registry.provenance)

    @staticmethod
    def overview(record):
        return ModelOverview(**{name: getattr(record, name) for name in ModelOverview.model_fields})
