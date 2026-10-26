import importlib
import pkgutil

import pytest

import riskbyte_ml
from riskbyte_api.services.inference import InferenceService
from riskbyte_ml.abrf import ABRFConfig


def test_all_ml_modules_and_inference_contract_import_without_artifacts():
    imported = {
        module.name
        for module in pkgutil.walk_packages(riskbyte_ml.__path__, riskbyte_ml.__name__ + ".")
        if importlib.import_module(module.name)
    }
    assert {
        "riskbyte_ml.preprocessing",
        "riskbyte_ml.features",
        "riskbyte_ml.models",
        "riskbyte_ml.evaluation",
        "riskbyte_ml.anomaly",
        "riskbyte_ml.behavioural",
        "riskbyte_ml.abrf",
        "riskbyte_ml.explainability",
    } <= imported
    assert InferenceService.__name__ == "InferenceService"


def test_abrf_weights_are_configurable_and_versioned():
    config = ABRFConfig(0.5, 0.3, 0.2, version="experimental-test-v1")
    assert config.classifier_weight == 0.5
    assert config.version == "experimental-test-v1"
    assert ABRFConfig().version == "provisional-v0"


@pytest.mark.parametrize(
    "weights",
    [(-0.1, 0.6, 0.5), (0.5, 0.5, 0.5), (float("nan"), 0.5, 0.5), (float("inf"), 0, 0)],
)
def test_abrf_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        ABRFConfig(*weights)


def test_abrf_requires_configuration_version():
    with pytest.raises(ValueError, match="version"):
        ABRFConfig(version=" ")
