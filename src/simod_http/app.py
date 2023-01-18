import uuid
from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd
from pydantic import BaseModel, BaseSettings

from simod.configuration import Configuration


class Exceptions:
    class InvalidConfig(Exception):
        pass

    class NotFound(Exception):
        pass

    class BadMultipartRequest(Exception):
        pass

    class UnsupportedMediaType(Exception):
        pass

    class InvalidCallbackUrl(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class DiscoveryFailed(Exception):
        pass


class Error(BaseModel):
    message: str


class RequestStatus(str, Enum):
    ACCEPTED = 'accepted'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class Response(BaseModel):
    request_id: str
    status: RequestStatus
    error: Union[Error, None]
    result_url: Union[str, None]


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
    storage_expire_after: int = 60 * 60 * 24 * 7  # 7 days

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

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f'Request(' \
               f'id={self.id}, ' \
               f'output_dir={self.output_dir}, ' \
               f'configuration={self.configuration}, ' \
               f'event_log_csv_path={self.event_log_csv_path}, ' \
               f'archive_url={self.archive_url}, ' \
               f'callback_endpoint={self.callback_endpoint})'

    def save(self):
        request_info_path = self.output_dir / 'request.json'
        request_info_path.write_text(self.json(exclude={'event_log': True}))

    @staticmethod
    def load(request_id: str, settings: Settings) -> 'Request':
        request_dir = Path(settings.simod_http_storage_path) / 'requests' / request_id
        request_info_path = request_dir / 'request.json'
        request_info = Request.parse_raw(request_info_path.read_text())
        return request_info

    @staticmethod
    def empty(storage_path: Path) -> 'Request':
        request_id = str(uuid.uuid4())

        output_dir = storage_path / 'requests' / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return Request(id=request_id, output_dir=output_dir)

    @staticmethod
    def make(
            storage_path: Path,
            configuration: Configuration,
            event_log: pd.DataFrame,
            callback_endpoint: Union[str, None] = None
    ) -> 'Request':
        request_id = str(uuid.uuid4())

        output_dir = storage_path / 'requests' / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return Request(
            id=request_id,
            output_dir=output_dir,
            configuration=configuration,
            event_log=event_log,
            callback_endpoint=callback_endpoint,
        )


settings = Settings()
settings.simod_http_storage_path = Path(settings.simod_http_storage_path)
