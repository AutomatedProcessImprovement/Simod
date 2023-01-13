import logging
import uuid
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Union

from pydantic import BaseModel, BaseSettings

from simod.configuration import Configuration


class Exceptions:
    class InvalidConfig(Exception):
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
    configuration: Configuration
    callback_endpoint: Union[str, None]

    @staticmethod
    def from_configuration(configuration: Configuration, callback_endpoint: Union[str, None] = None) -> 'Request':
        return Request(
            id=uuid.uuid4().hex,
            configuration=configuration,
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
