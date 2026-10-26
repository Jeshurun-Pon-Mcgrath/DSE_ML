"""Versioned model evaluation reports; no transaction prediction endpoint."""

from fastapi import APIRouter, Request

from riskbyte_ml.models.supervised.schemas import (
    ModelComparison,
    ModelMetricView,
    ModelOverview,
    ModelRecord,
    ModelResponse,
    ScenarioRecallView,
    ThresholdAnalysis,
)

router = APIRouter(prefix="/api/v1/models", tags=["supervised evaluation"])


@router.get("", response_model=ModelResponse[list[ModelOverview]])
def models(request: Request):
    return request.app.state.model_service.response("list")


@router.get("/comparison", response_model=ModelResponse[ModelComparison])
def comparison(request: Request):
    return request.app.state.model_service.response("comparison")


@router.get("/selected", response_model=ModelResponse[ModelRecord])
def selected(request: Request):
    return request.app.state.model_service.response("selected")


@router.get("/{model_id}/metrics", response_model=ModelResponse[ModelMetricView])
def metrics(request: Request, model_id: str):
    return request.app.state.model_service.response("metrics", model_id)


@router.get("/{model_id}/scenario-recall", response_model=ModelResponse[ScenarioRecallView])
def scenarios(request: Request, model_id: str):
    return request.app.state.model_service.response("scenarios", model_id)


@router.get("/{model_id}/threshold-analysis", response_model=ModelResponse[ThresholdAnalysis])
def thresholds(request: Request, model_id: str):
    return request.app.state.model_service.response("thresholds", model_id)
