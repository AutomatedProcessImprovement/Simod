import uuid
from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd
from pydantic import BaseModel, BaseSettings

from simod.configuration import Configuration


class Error(BaseModel):
    message: str


class RequestStatus(str, Enum):
    UNKNOWN = 'unknown'
    ACCEPTED = 'accepted'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class Response(BaseModel):
    request_id: str
    status: RequestStatus
    error: Union[Error, None]
    archive_url: Union[str, None]


class Settings(BaseSettings):
    """
    Application settings.
    """

    # These host and port are used to compose a link to the resulting archive.
    simod_http_host: str = "localhost"
    simod_http_port: int = 8000
    simod_http_scheme: str = "http"

    # Path on the file system to store results until the user fetches them, or they expire.
    simod_http_storage_path: str = "/tmp/simod"
    simod_http_request_expiration_timedelta: int = 60 * 60 * 24 * 7  # 7 days
    simod_http_storage_cleaning_timedelta: int = 60

    # Logging levels: CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET
    simod_http_logging_level: str = "debug"
    simod_http_logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    simod_http_log_path: Union[str, None] = None

    class Config:
        env_file = ".env"

    def __init__(self, **data):
        super().__init__(**data)

        storage_path = Path(self.simod_http_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)


class Request(BaseModel):
    id: str
    output_dir: Path
    status: Union[RequestStatus, None] = None
    configuration: Union[Configuration, None] = None
    event_log: Union[pd.DataFrame, None] = None  # NOTE: this field isn't present in request.json
    event_log_csv_path: Union[Path, None] = None
    callback_endpoint: Union[str, None] = None
    archive_url: Union[str, None] = None
    timestamp: Union[pd.Timestamp, None] = None

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f'Request(' \
               f'id={self.id}, ' \
               f'output_dir={self.output_dir}, ' \
               f'configuration={self.configuration}, ' \
               f'event_log_csv_path={self.event_log_csv_path}, ' \
               f'archive_url={self.archive_url}, ' \
               f'timestamp={self.timestamp}, ' \
               f'callback_endpoint={self.callback_endpoint})'

    def save(self):
        request_info_path = self.output_dir / 'request.json'
        request_info_path.write_text(self.json(exclude={'event_log': True}))

    @staticmethod
    def load(request_id: str, settings: Settings) -> 'Request':
        request_dir = Path(settings.simod_http_storage_path) / 'requests' / request_id
        if not request_dir.exists():
            raise NotFound(
                request_id=request_id,
                status=RequestStatus.UNKNOWN,
                archive_url=None,
                message=f'Request {request_id} not found on the server',
            )

        try:
            request_info_path = request_dir / 'request.json'
            request_info = Request.parse_raw(request_info_path.read_text())
            return request_info

        except Exception as e:
            raise InternalServerError(
                request_id=request_id,
                status=RequestStatus.UNKNOWN,
                archive_url=None,
                message=f'Failed to load request {request_id}: {e}',
            )

    @staticmethod
    def empty(storage_path: Path) -> 'Request':
        request_id = str(uuid.uuid4())

        output_dir = storage_path / 'requests' / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return Request(
            id=request_id,
            output_dir=output_dir,
            status=RequestStatus.UNKNOWN,
            configuration=None,
            event_log=None,
            event_log_csv_path=None,
            callback_endpoint=None,
            archive_url=None,
            timestamp=None,
        )


class BaseRequestException(Exception):

    def __init__(self, request_id: str, message: str, status: RequestStatus, archive_url: Union[str, None] = None):
        self.request_id = request_id
        self.status = status
        self.archive_url = archive_url
        self.message = message

    def make_response(self) -> Response:
        return Response(
            request_id=self.request_id,
            status=self.status,
            archive_url=self.archive_url,
            error=Error(message=self.message),
        )


class NotFound(BaseRequestException):
    pass


class BadMultipartRequest(BaseRequestException):
    pass


class UnsupportedMediaType(BaseRequestException):
    pass


class InternalServerError(BaseRequestException):
    pass
