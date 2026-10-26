"""Safe domain errors; never include filesystem paths or source values in messages."""


class DatasetError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
