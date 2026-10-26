"""Read-only offline research explanations; never a live prediction endpoint."""

from fastapi import APIRouter, Request
from pydantic import JsonValue

from riskbyte_ml.explainability.schemas import ExplanationResponse, UnifiedExplanation

router = APIRouter(prefix="/api/v1/explainability", tags=["Explainability research"])


@router.get("/status", response_model=ExplanationResponse[dict[str, JsonValue]])
def status(request: Request):
    return request.app.state.explainability_service.response("status")


@router.get("/global", response_model=ExplanationResponse[dict[str, JsonValue]])
def global_importance(request: Request):
    return request.app.state.explainability_service.response("global")


@router.get("/families", response_model=ExplanationResponse[dict[str, JsonValue]])
def families(request: Request):
    return request.app.state.explainability_service.response("families")


@router.get("/case-studies", response_model=ExplanationResponse[dict[str, JsonValue]])
def case_studies(request: Request):
    return request.app.state.explainability_service.response("case-studies")


@router.get("/validation-errors", response_model=ExplanationResponse[dict[str, JsonValue]])
def validation_errors(request: Request):
    return request.app.state.explainability_service.response("validation-errors")


@router.get("/validation/{transaction_id}", response_model=ExplanationResponse[UnifiedExplanation])
def validation_transaction(transaction_id: str, request: Request):
    return request.app.state.explainability_service.response("validation", transaction_id)
