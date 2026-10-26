"""Read-only anomaly and behavioural analysis endpoints."""

from fastapi import APIRouter, Request
from pydantic import JsonValue

from riskbyte_ml.anomaly.schemas import DetectionMetrics, Phase5Response

router = APIRouter(prefix="/api/v1", tags=["anomaly and behavioural analysis"])


@router.get("/anomaly/status", response_model=Phase5Response[dict[str, JsonValue]])
def status(request: Request):
    return request.app.state.phase5_service.response("status")


@router.get("/anomaly/metrics", response_model=Phase5Response[DetectionMetrics])
def metrics(request: Request):
    return request.app.state.phase5_service.response("metrics")


@router.get("/anomaly/scenarios", response_model=Phase5Response[dict[str, JsonValue]])
def scenarios(request: Request):
    return request.app.state.phase5_service.response("scenarios")


@router.get("/anomaly/hard-negatives", response_model=Phase5Response[dict[str, JsonValue]])
def hard_negatives(request: Request):
    return request.app.state.phase5_service.response("hard-negatives")


@router.get("/behaviour/summary", response_model=Phase5Response[dict[str, JsonValue]])
def summary(request: Request):
    return request.app.state.phase5_service.response("summary")


@router.get("/behaviour/components", response_model=Phase5Response[dict[str, JsonValue]])
def components(request: Request):
    return request.app.state.phase5_service.response("components")


@router.get("/behaviour/validation-analysis", response_model=Phase5Response[dict[str, JsonValue]])
def validation_analysis(request: Request):
    return request.app.state.phase5_service.response("validation-analysis")
