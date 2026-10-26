from pydantic import BaseModel


class ValidationIssue(BaseModel):
    location: list[str | int]
    code: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    issues: list[ValidationIssue] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
