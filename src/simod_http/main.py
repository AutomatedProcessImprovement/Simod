import logging
import shutil
from pathlib import Path
from typing import Callable, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, Response, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi_utils.tasks import repeat_every
from pydantic import ValidationError
from uvicorn.config import LOGGING_CONFIG

from simod.configuration import Configuration
from simod.event_log.utilities import read as read_event_log
from simod_http.app import Response as AppResponse, RequestStatus, Request as AppRequest, Settings, NotFound, \
    BadMultipartRequest, UnsupportedMediaType, BaseRequestException, Error
from simod_http.archiver import make_url_for
from simod_http.executor import Executor

settings = Settings()
settings.simod_http_storage_path = Path(settings.simod_http_storage_path)

app = FastAPI()


def run_simod_discovery(request: Request, settings: Settings):
    """
    Run Simod with the user's configuration.
    """
    executor = Executor(app_settings=settings, request=request)
    executor.run()


class CustomRequest(Request):
    async def body(self) -> list[bytes]:
        if not hasattr(self, '_body'):
            body = await super().body()

            # If the request multipart, then the body is a list of bytes split by the boundary specified in the header
            content_type = self.headers.get('content-type', '')
            if 'multipart/mixed' in content_type or 'multipart/form-data' in content_type:
                boundary = self.headers['content-type'].split('boundary=')[1]
                boundary = boundary.strip('"')
                body = body.split(f'--{boundary}\n'.encode())[1:]

            self._body = body

        return self._body


class CustomRouteMatcher(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                request = CustomRequest(request.scope, request.receive)
                return await original_route_handler(request)
            except Exception as e:
                if isinstance(e, BaseRequestException):
                    return e.json_response()
                elif isinstance(e, ValidationError):
                    err = Error(message=str(e), details=e.errors())
                elif isinstance(e, HTTPException):
                    err = Error(message=e.detail)
                else:
                    err = Error(message=str(e), details=repr(e))

                response = AppResponse.construct()
                response.error = err
                return response.json_response(status_code=500)

        return custom_route_handler


app.router.route_class = CustomRouteMatcher


@app.on_event('startup')
async def application_startup():
    logging_handlers = []
    if settings.simod_http_log_path is not None:
        logging_handlers.append(logging.FileHandler(settings.simod_http_log_path, mode='w'))

    if len(logging_handlers) > 0:
        logging.basicConfig(
            level=settings.simod_http_logging_level.upper(),
            handlers=logging_handlers,
            format=settings.simod_http_logging_format,
        )
    else:
        logging.basicConfig(
            level=settings.simod_http_logging_level.upper(),
            format=settings.simod_http_logging_format,
        )

    logging.debug(f'Application settings: {settings}')


@app.on_event('shutdown')
async def application_shutdown():
    requests_dir = Path(settings.simod_http_storage_path) / 'requests'
    for request_dir in requests_dir.iterdir():
        try:
            request = AppRequest.load(request_dir.name, settings)
        except Exception as e:
            logging.error(f'Failed to load request: {request_dir.name}, {str(e)}')
            continue

        # Sets 'running' requests to 'failure'
        if request.status == RequestStatus.RUNNING:
            request.status = RequestStatus.FAILURE
            request.timestamp = pd.Timestamp.now()
            request.save()


@app.on_event('startup')
@repeat_every(seconds=settings.simod_http_storage_cleaning_timedelta)
async def clean_up():
    requests_dir = Path(settings.simod_http_storage_path) / 'requests'
    current_timestamp = pd.Timestamp.now()
    expire_after_delta = pd.Timedelta(seconds=settings.simod_http_request_expiration_timedelta)
    for request_dir in requests_dir.iterdir():
        if request_dir.is_dir():
            logging.info(f'Checking request directory for expired data: {request_dir}')

            # Removes empty directories
            if len(list(request_dir.iterdir())) == 0:
                logging.info(f'Removing empty directory: {request_dir}')
                shutil.rmtree(request_dir, ignore_errors=True)

            # Removes orphaned request directories
            if not (request_dir / 'request.json').exists():
                logging.info(f'Removing request folder for {request_dir.name}, no request.json file')
                shutil.rmtree(request_dir, ignore_errors=True)

            try:
                request = AppRequest.load(request_dir.name, settings)
            except Exception as e:
                logging.error(f'Failed to load request: {request_dir.name}, {str(e)}')
                continue

            # Removes expired requests
            expired_at = request.timestamp + expire_after_delta
            if expired_at <= current_timestamp:
                logging.info(f'Removing request folder for {request_dir.name}, expired at {expired_at}')
                shutil.rmtree(request_dir, ignore_errors=True)

            # Removes requests without timestamp that are not running
            if request.timestamp is None and request.status != RequestStatus.RUNNING:
                logging.info(f'Removing request folder for {request_dir.name}, no timestamp and not running')
                shutil.rmtree(request_dir, ignore_errors=True)


@app.exception_handler(BaseRequestException)
async def request_exception_handler(_, exc: BaseRequestException) -> JSONResponse:
    logging.error(f'Request exception occurred: {exc}')
    return exc.json_response()


@app.get('/discoveries/{request_id}/{file_name}')
async def read_discovery_file(request_id: str, file_name: str):
    """
    Get a file from a discovery request.
    """
    request = AppRequest.load(request_id, settings)

    if not request.output_dir.exists():
        raise NotFound(request_id=request_id, request_status=request.status, message='Request not found on the server')

    file_path = request.output_dir / file_name
    if not file_path.exists():
        raise NotFound(request_id=request_id, request_status=request.status, message=f'File not found: {file_name}')

    media_type = await _infer_media_type_from_extension(file_name)

    return Response(
        content=file_path.read_bytes(),
        media_type=media_type,
        headers={
            'Content-Disposition': f'attachment; filename="{file_name}"',
        }
    )


@app.get('/discoveries/{request_id}')
async def read_discovery(request_id: str) -> AppResponse:
    """
    Get the status of the request.
    """
    request = AppRequest.load(request_id, settings)

    return AppResponse(
        request_id=request_id,
        request_status=request.status,
        archive_url=make_url_for(request.id, Path(f'{request.id}.tar.gz'), settings),
    )


@app.post('/discoveries')
async def create_discovery(
        background_tasks: BackgroundTasks,
        bodies: list[bytes] = Body(),
) -> JSONResponse:
    """
    Create a new business process model discovery and optimization request.
    """
    global settings

    request = AppRequest.empty(Path(settings.simod_http_storage_path))

    configuration, event_log, event_log_csv_path = _parse_bodies_from_request(
        request.id,
        request.status,
        bodies,
        request.output_dir,
    )

    if event_log is None:
        raise NotFound(
            request_id=request.id,
            request_status=request.status,
            archive_url=request.archive_url,
            message='Event log not found',
        )

    request.configuration = configuration
    request.event_log = event_log
    request.event_log_csv_path = event_log_csv_path
    request.status = RequestStatus.ACCEPTED
    request.save()

    response = AppResponse(request_id=request.id, request_status=request.status)

    background_tasks.add_task(run_simod_discovery, request, settings)

    return response.json_response(status_code=202)


@app.get('/{any_str}')
async def root() -> JSONResponse:
    raise NotFound(
        request_id='N/A',
        request_status=RequestStatus.UNKNOWN,
        message='Not found',
    )


def _parse_bodies_from_request(
        request_id: str,
        request_status: RequestStatus,
        files: list[bytes],
        output_dir: Path,
) -> tuple[Configuration, Union[pd.DataFrame, None], Union[Path, None]]:
    configuration = None
    event_log = None
    event_log_file_extension = None
    bpmn_model = None

    for content in files:
        parts = content.split(b'\n\n')
        if len(parts) >= 2:
            header = _parse_headers(parts[0])
            body = parts[1]
        else:
            raise BadMultipartRequest(
                request_id=request_id,
                request_status=request_status,
                message='Each part of the multipart request must have a header and a body',
            )

        filename = _get_multipart_filename_from_header(header)

        if _is_header_yaml(header.get('content-type')) \
                or '.yaml' in filename \
                or '.yml' in filename:
            configuration = Configuration.from_stream(body)
            continue

        if '.bpmn' in filename:
            bpmn_model = body
            continue

        event_log = body
        event_log_file_extension = _infer_event_log_file_extension_from_header(
            request_id,
            request_status,
            header.get('content-type')
        )

    if configuration is None:
        configuration = Configuration.default()

    if bpmn_model is not None:
        bpmn_path = output_dir / f'model.bpmn'
        bpmn_path.write_bytes(bpmn_model)

        configuration.common.model_path = bpmn_path.absolute()

    csv_path = None

    if event_log is not None:
        event_log_path = output_dir / f'event_log{event_log_file_extension}'
        event_log_path.write_bytes(event_log)

        configuration.common.log_path = event_log_path.absolute()
        configuration.common.test_log_path = None

        event_log, csv_path = read_event_log(configuration.common.log_path, configuration.common.log_ids)

    return configuration, event_log, csv_path


def _parse_headers(header: bytes) -> dict[str, str]:
    headers = {}
    for line in header.split(b'\n'):
        if line:
            key, value = line.split(b':')
            key = key.decode('utf-8').strip().lower()
            value = value.decode('utf-8').strip().rstrip(';').lower()
            if ';' in value:
                value_parts = value.split(';')
                value = []
                for value_part in value_parts:
                    if '=' in value_part:
                        value_part_key, value_part_value = value_part.split('=')
                        value_part_key = value_part_key.strip()
                        value_part_value = value_part_value.strip().strip('"')
                        value.append({value_part_key: value_part_value})
                    else:
                        value.append({'value': value_part})
            headers[key] = value
    return headers


def _get_multipart_filename_from_header(header: dict[str, str]) -> str:
    content = header.get('content-disposition', [])

    if len(content) == 0:
        return ''

    names = list(
        map(lambda x: x.get('filename', ''),
            filter(lambda x: 'filename' in x, content))
    )

    if len(names) == 0:
        return ''

    return names[0]


def _is_header_yaml(header: str) -> bool:
    return 'application/x-yaml' in header or \
        'application/yaml' in header or \
        'text/yaml' in header or \
        'text/x-yaml' in header or \
        'text/vnd.yaml' in header


def _is_header_xml(header: str) -> bool:
    return 'application/xml' in header or \
        'text/xml' in header or \
        'text/x-xml' in header or \
        'text/vnd.xml' in header


def _infer_event_log_file_extension_from_header(
        request_id: str,
        request_status: RequestStatus,
        content_type: str,
) -> str:
    if 'text/csv' in content_type:
        return '.csv'
    elif 'application/xml' in content_type or 'text/xml' in content_type:
        return '.xml'
    else:
        raise UnsupportedMediaType(
            request_id=request_id,
            request_status=request_status,
            archive_url=None,
            message='Unsupported event log file type',
        )


async def _infer_media_type_from_extension(file_name) -> str:
    if file_name.endswith('.csv'):
        media_type = 'text/csv'
    elif file_name.endswith('.xml'):
        media_type = 'application/xml'
    elif file_name.endswith('.xes'):
        media_type = 'application/xml'
    elif file_name.endswith('.bpmn'):
        media_type = 'application/xml'
    elif file_name.endswith('.json'):
        media_type = 'application/json'
    elif file_name.endswith('.png'):
        media_type = 'image/png'
    elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        media_type = 'image/jpeg'
    elif file_name.endswith('.pdf'):
        media_type = 'application/pdf'
    elif file_name.endswith('.txt'):
        media_type = 'text/plain'
    elif file_name.endswith('.zip'):
        media_type = 'application/zip'
    elif file_name.endswith('.gz'):
        media_type = 'application/gzip'
    elif file_name.endswith('.tar'):
        media_type = 'application/tar'
    elif file_name.endswith('.tar.gz'):
        media_type = 'application/tar+gzip'
    elif file_name.endswith('.tar.bz2'):
        media_type = 'application/x-bzip2'
    else:
        media_type = 'application/octet-stream'

    return media_type


if __name__ == '__main__':
    logging_config = LOGGING_CONFIG
    logging_config['formatters']['default']['fmt'] = settings.simod_http_logging_format
    logging_config['formatters']['access']['fmt'] = settings.simod_http_logging_format.replace(
        '%(message)s', '%(client_addr)s - "%(request_line)s" %(status_code)s')

    uvicorn.run(
        'main:app',
        host=settings.simod_http_host,
        port=settings.simod_http_port,
        log_level='info',
        log_config=logging_config,
    )
