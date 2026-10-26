"""Bounded typed statistical summaries; no analysis or training in HTTP handlers."""

from fastapi import APIRouter, Request

from riskbyte_ml.statistics.models import (
    CorrelationReport,
    HardNegativeReport,
    HypothesisReport,
    PrevalenceReport,
    ScenarioReport,
    StatisticsResponse,
    StatisticsSummary,
    TemporalReport,
)

router = APIRouter(prefix="/api/v1/statistics", tags=["statistical intelligence"])


@router.get("/summary", response_model=StatisticsResponse[StatisticsSummary])
def summary(request: Request):
    return request.app.state.statistics_service.response("summary")


@router.get("/hypotheses", response_model=StatisticsResponse[HypothesisReport])
def hypotheses(request: Request):
    return request.app.state.statistics_service.response("hypotheses")


@router.get("/correlations", response_model=StatisticsResponse[CorrelationReport])
def correlations(request: Request):
    return request.app.state.statistics_service.response("correlations")


@router.get("/prevalence", response_model=StatisticsResponse[PrevalenceReport])
def prevalence(request: Request):
    return request.app.state.statistics_service.response("prevalence")


@router.get("/scenarios", response_model=StatisticsResponse[ScenarioReport])
def scenarios(request: Request):
    return request.app.state.statistics_service.response("scenarios")


@router.get("/hard-negatives", response_model=StatisticsResponse[HardNegativeReport])
def hard_negatives(request: Request):
    return request.app.state.statistics_service.response("hard_negatives")


@router.get("/temporal", response_model=StatisticsResponse[TemporalReport])
def temporal(request: Request):
    return request.app.state.statistics_service.response("temporal")
