import logging
import uuid
from enum import Enum
from logging import Logger
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


class Settings(BaseSettings):
    """
    Application settings.
    """

    # These host and port are used to compose a link to the resulting archive.
    external_host: str = "localhost"
    external_port: int = 8000
    external_scheme: str = "http"

    # Path on the file system to store results until the user fetches them, or they expire.
    storage_path: str = "/tmp/simod"
    storage_expire_after: int = 60 * 60 * 24 * 7  # 7 days

    # Logging levels: CRITICAL FATAL ERROR WARNING WARN INFO DEBUG NOTSET
    logging_level: str = "debug"
    log_path: Union[str, None] = None
    logger: Logger = logging.getLogger(__name__)

    class Config:
        env_file = ".env"

    def __init__(self, **data):
        super().__init__(**data)

        storage_path = Path(self.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)


class Request(BaseModel):
    id: str
    request_dir: Path
    configuration: Union[Configuration, None] = None
    event_log: Union[pd.DataFrame, None] = None
    event_log_csv_path: Union[Path, None] = None
    callback_endpoint: Union[str, None] = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def empty(storage_path: Path) -> 'Request':
        request_id = str(uuid.uuid4())

        request_dir = storage_path / 'requests' / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        return Request(id=request_id, request_dir=request_dir)

    @staticmethod
    def make(
            storage_path: Path,
            configuration: Configuration,
            event_log: pd.DataFrame,
            callback_endpoint: Union[str, None] = None
    ) -> 'Request':
        request_id = str(uuid.uuid4())

        request_dir = storage_path / 'requests' / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        return Request(
            id=request_id,
            request_dir=request_dir,
            configuration=configuration,
            event_log=event_log,
            callback_endpoint=callback_endpoint,
        )


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


settings = Settings()
settings.storage_path = Path(settings.storage_path)
