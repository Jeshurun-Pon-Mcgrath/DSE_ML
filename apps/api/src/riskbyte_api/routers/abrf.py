"""Versioned read-only ABRF research routes; no public prediction surface."""

from fastapi import APIRouter, Request
from pydantic import JsonValue

from riskbyte_ml.abrf.contracts import ABRFResponse

router = APIRouter(prefix="/api/v1/abrf", tags=["ABRF development research"])


@router.get("/status", response_model=ABRFResponse[dict[str, JsonValue]])
def status(request: Request):
    return request.app.state.abrf_service.response("status")


@router.get("/config", response_model=ABRFResponse[dict[str, JsonValue]])
def config(request: Request):
    return request.app.state.abrf_service.response("config")


@router.get("/validation", response_model=ABRFResponse[dict[str, JsonValue]])
def validation(request: Request):
    return request.app.state.abrf_service.response("validation")


@router.get("/ablation", response_model=ABRFResponse[dict[str, JsonValue]])
def ablation(request: Request):
    return request.app.state.abrf_service.response("ablation")


@router.get("/scenarios", response_model=ABRFResponse[dict[str, JsonValue]])
def scenarios(request: Request):
    return request.app.state.abrf_service.response("scenarios")


@router.get("/hard-negatives", response_model=ABRFResponse[dict[str, JsonValue]])
def hard_negatives(request: Request):
    return request.app.state.abrf_service.response("hard-negatives")
