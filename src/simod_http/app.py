import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Union, Any, Optional

import pandas as pd
import pika
from fastapi.responses import JSONResponse
from pika.spec import PERSISTENT_DELIVERY_MODE
from pydantic import BaseModel, BaseSettings


class Error(BaseModel):
    message: str
    details: Union[Any, None] = None


class RequestStatus(str, Enum):
    UNKNOWN = 'unknown'
    ACCEPTED = 'accepted'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class NotificationMethod(str, Enum):
    HTTP = 'callback'
    EMAIL = 'email'


class NotificationSettings(BaseModel):
    method: Union[NotificationMethod, None] = None
    callback_url: Union[str, None] = None
    email: Union[str, None] = None


class Response(BaseModel):
    request_id: str
    request_status: RequestStatus
    error: Union[Error, None]
    archive_url: Union[str, None]

    def json_response(self, status_code: int) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=self.dict(),
        )


class Request(BaseModel):
    id: str
    output_dir: Path
    status: Union[RequestStatus, None] = None
    configuration_path: Union[Path, None] = None
    archive_url: Union[str, None] = None
    timestamp: Union[pd.Timestamp, None] = None
    notification_settings: Union[NotificationSettings, None] = None
    notified: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f'Request(' \
               f'id={self.id}, ' \
               f'output_dir={self.output_dir}, ' \
               f'status={self.status}, ' \
               f'configuration_path={self.configuration_path}, ' \
               f'archive_url={self.archive_url}, ' \
               f'timestamp={self.timestamp}, ' \
               f'notification_settings={self.notification_settings}, ' \
               f'notified={self.notified})'

    def save(self):
        request_info_path = self.output_dir / 'request.json'
        request_info_path.write_text(self.json(exclude={'event_log': True}))

    @staticmethod
    def empty(storage_path: Path) -> 'Request':
        request_id = str(uuid.uuid4())

        output_dir = storage_path / 'requests' / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return Request(
            id=request_id,
            output_dir=output_dir.absolute(),
            status=RequestStatus.UNKNOWN,
            configuration_path=None,
            callback_endpoint=None,
            archive_url=None,
            timestamp=pd.Timestamp.now(),
        )


class Application(BaseSettings):
    """
    Simod application that stores main settings and provides access to internal API.
    """

    # These host and port are used to compose a link to the resulting archive.
    simod_http_host: str = 'localhost'
    simod_http_port: int = 8000
    simod_http_scheme: str = 'http'

    # Path on the file system to store results until the user fetches them, or they expire.
    simod_http_storage_path: str = '/tmp/simod'
    simod_http_request_expiration_timedelta: int = 60 * 60 * 24 * 7  # 7 days
    simod_http_storage_cleaning_timedelta: int = 60

    # Logging levels: CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET
    simod_http_logging_level: str = 'debug'
    simod_http_logging_format = '%(asctime)s \t %(name)s \t %(levelname)s \t %(message)s'
    simod_http_log_path: Union[str, None] = None

    # SMTP server settings
    simod_http_smtp_server: str = 'localhost'
    simod_http_smtp_port: int = 25

    # Queue settings
    simod_http_requests_queue_name: str = 'requests'
    simod_http_results_queue_name: str = 'results'
    rabbitmq_url: str = 'amqp://guest:guest@localhost:5672/'

    class Config:
        env_file = ".env"

    def __init__(self, **data):
        super().__init__(**data)

        storage_path = Path(self.simod_http_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def init() -> 'Application':
        debug = os.environ.get('SIMOD_HTTP_DEBUG', 'false').lower() == 'true'

        if debug:
            app = Application()
        else:
            app = Application(_env_file='.env.production')

        app.simod_http_storage_path = Path(app.simod_http_storage_path)

        return app

    def load_request(self, request_id: str) -> Request:
        request_dir = Path(self.simod_http_storage_path) / 'requests' / request_id
        if not request_dir.exists():
            raise NotFound(
                request_id=request_id,
                request_status=RequestStatus.UNKNOWN,
                archive_url=None,
                message='Request is not found on the server',
            )

        request_info_path = request_dir / 'request.json'
        request = Request.parse_raw(request_info_path.read_text())
        return request

    def new_request_from_params(self, callback_url: Optional[str] = None, email: Optional[str] = None) -> 'Request':
        request = Request.empty(Path(self.simod_http_storage_path))

        if callback_url is not None:
            notification_settings = NotificationSettings(
                method=NotificationMethod.HTTP,
                callback_url=callback_url,
            )
        elif email is not None:
            notification_settings = NotificationSettings(
                method=NotificationMethod.EMAIL,
                email=email,
            )
        else:
            notification_settings = None

        request.notification_settings = notification_settings

        return request

    def make_results_url_for(self, request: Request) -> Union[str, None]:
        if request.status == RequestStatus.SUCCESS:
            if self.simod_http_port == 80:
                port = ''
            else:
                port = f':{self.simod_http_port}'
            return f'{self.simod_http_scheme}://{self.simod_http_host}{port}' \
                   f'/discoveries' \
                   f'/{request.id}' \
                   f'/{request.id}.tar.gz'
        return None

    def publish_request(self, request: Request):
        parameters = pika.URLParameters(self.rabbitmq_url)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=self.simod_http_requests_queue_name, durable=True)
        channel.basic_publish(
            exchange='',
            routing_key=self.simod_http_requests_queue_name,
            body=request.id.encode(),
            properties=pika.BasicProperties(
                delivery_mode=PERSISTENT_DELIVERY_MODE,
                content_type='text/plain',
            ),
        )
        connection.close()
        logging.info(f'Published request {request.id} to the {self.simod_http_requests_queue_name} queue')


class BaseRequestException(Exception):
    _status_code = 500

    def __init__(
            self,
            request_id: str,
            message: str,
            request_status: RequestStatus,
            archive_url: Union[str, None] = None,
    ):
        self.request_id = request_id
        self.request_status = request_status
        self.archive_url = archive_url
        self.message = message

    @property
    def status_code(self) -> int:
        return self._status_code

    def make_response(self) -> Response:
        return Response(
            request_id=self.request_id,
            request_status=self.request_status,
            archive_url=self.archive_url,
            error=Error(message=self.message),
        )

    def json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=self.make_response().dict(),
        )


class NotFound(BaseRequestException):
    _status_code = 404


class BadMultipartRequest(BaseRequestException):
    _status_code = 400


class UnsupportedMediaType(BaseRequestException):
    _status_code = 415


class InternalServerError(BaseRequestException):
    _status_code = 500


class NotSupported(BaseRequestException):
    _status_code = 501


app = Application.init()
