import logging
import shutil
from pathlib import Path
from typing import Callable, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, Response, Body
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi_utils.tasks import repeat_every

from simod.configuration import Configuration
from simod.event_log.utilities import read as read_event_log
from simod_http.app import Response as AppResponse, RequestStatus, Request as AppRequest, Settings, NotFound, \
    BadMultipartRequest, UnsupportedMediaType, InternalServerError
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


class MultiPartRequest(Request):
    async def body(self) -> list[bytes]:
        if not hasattr(self, '_body'):
            body = await super().body()
            if "multipart/mixed" in self.headers.get("content-type", ""):
                boundary = self.headers["content-type"].split("boundary=")[1]
                boundary = boundary.strip('"')
                body = body.split(f'--{boundary}\n'.encode())[1:]
            self._body = body
        return self._body


class MultiPartRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = MultiPartRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


app.router.route_class = MultiPartRoute


@app.on_event('startup')
async def application_startup():
    logging_handlers = []
    if settings.simod_http_log_path is not None:
        logging_handlers.append(logging.FileHandler(settings.simod_http_log_path, mode='w'))

    logging.basicConfig(
        level=settings.simod_http_logging_level.upper(),
        handlers=logging_handlers,
        format=settings.simod_http_logging_format,
    )

    logging.debug(f'Application settings: {settings}')


@app.on_event('startup')
@repeat_every(seconds=settings.simod_http_storage_cleaning_timedelta)
async def clean_up():
    requests_dir = Path(settings.simod_http_storage_path) / 'requests'
    current_timestamp = pd.Timestamp.now()
    expire_after_delta = pd.Timedelta(seconds=settings.simod_http_request_expiration_timedelta)
    for request_dir in requests_dir.iterdir():
        if request_dir.is_dir():
            logging.info(f'Checking request directory for expired data: {request_dir}')
            try:
                request = AppRequest.load(request_dir.name, settings)
            except Exception as e:
                logging.error(f'Failed to load request: {e}')
                continue

            # Removes expired requests
            expired_at = request.timestamp + expire_after_delta
            if expired_at <= current_timestamp:
                logging.info(f'Removing request folder for {request_dir.name}, expired at {expired_at}')
                shutil.rmtree(request_dir, ignore_errors=True)

            # Removes orphaned request directories
            if not (request_dir / 'request.json').exists():
                logging.info(f'Removing request folder for {request_dir.name}, no request.json file')
                shutil.rmtree(request_dir, ignore_errors=True)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logging.error(f'HTTP exception: {exc}')
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.exception_handler(BadMultipartRequest)
async def bad_multipart_request_exception_handler(request, exc: BadMultipartRequest):
    logging.error(f'{BadMultipartRequest.__class__}: {exc}')
    response = exc.make_response()
    return JSONResponse(status_code=400, content=response.dict())


@app.exception_handler(UnsupportedMediaType)
async def unsupported_media_type_exception_handler(request, exc: UnsupportedMediaType):
    logging.error(f'{UnsupportedMediaType.__class__}: {exc}')
    response = exc.make_response()
    return JSONResponse(status_code=415, content=response.dict())


@app.exception_handler(NotFound)
async def not_found_exception_handler(request, exc: NotFound):
    logging.error(f'{NotFound.__class__}: {exc}')
    response = exc.make_response()
    return JSONResponse(status_code=404, content=response.dict())


@app.exception_handler(InternalServerError)
async def internal_server_error_exception_handler(request, exc: InternalServerError):
    logging.error(f'{InternalServerError.__class__}: {exc}')
    response = exc.make_response()
    return JSONResponse(status_code=500, content=response.dict())


@app.get('/discoveries/{request_id}/{file_name}')
async def read_discovery_file(request_id: str, file_name: str):
    request = AppRequest.load(request_id, settings)

    if not request.output_dir.exists():
        raise HTTPException(status_code=404, detail='Request not found')

    file_path = request.output_dir / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail='File not found')

    media_type = 'application/tar+gzip'
    if file_name.endswith('.csv'):
        media_type = 'text/csv'

    return Response(
        content=file_path.read_bytes(),
        media_type=media_type,
        headers={
            'Content-Disposition': f'attachment; filename="{file_name}"',
        }
    )


@app.get('/discoveries/{request_id}',
         response_model=AppResponse,
         response_model_exclude_unset=True,
         response_model_exclude_none=True)
def read_discovery(request_id: str) -> AppResponse:
    """
    Get the status of the request.
    """
    request = AppRequest.load(request_id, settings)

    return AppResponse(
        request_id=request_id,
        status=request.status,
        archive_url=make_url_for(request.id, Path(f'{request.id}.tar.gz'), settings),
    )


@app.post('/discoveries',
          response_model=AppResponse,
          response_model_exclude_none=True,
          response_model_exclude_unset=True)
async def create_discovery(
        background_tasks: BackgroundTasks,
        files: list[bytes] = Body(),
) -> JSONResponse:
    """
    Create a new business process model discovery and optimization request.
    """
    global settings

    request = AppRequest.empty(Path(settings.simod_http_storage_path))

    configuration, event_log, event_log_csv_path = _parse_files_from_request(
        request.id,
        request.status,
        files,
        request.output_dir,
    )

    if event_log is None:
        raise NotFound(
            request_id=request.id,
            status=request.status,
            archive_url=request.archive_url,
            message='Event log not found',
        )

    request.configuration = configuration
    request.event_log = event_log
    request.event_log_csv_path = event_log_csv_path
    request.status = RequestStatus.ACCEPTED
    request.save()

    response = AppResponse(request_id=request.id, status=request.status)

    background_tasks.add_task(run_simod_discovery, request, settings)

    return JSONResponse(status_code=202, content=response.dict())


def _parse_files_from_request(
        request_id: str,
        request_status: RequestStatus,
        files: list[bytes],
        output_dir: Path,
) -> tuple[Configuration, Union[pd.DataFrame, None], Union[Path, None]]:
    configuration = None
    event_log = None
    event_log_file_extension = None

    for content in files:
        parts = content.split(b'\n\n')
        if len(parts) >= 2:
            header = parts[0]
            body = parts[1]
        else:
            raise BadMultipartRequest(
                request_id=request_id,
                status=request_status,
                archive_url=None,
                message='Each part of the multipart request must have a header and a body',
            )

        if _is_header_yaml(header):
            configuration = Configuration.from_stream(body)
            continue

        event_log = body
        event_log_file_extension = _infer_event_log_file_extension_from_header(request_id, request_status, header)

    if configuration is None:
        configuration = Configuration.default()

    # TODO: add BPMN file support

    # TODO: update model_path

    csv_path = None

    if event_log is not None:
        event_log_path = output_dir / f'event_log{event_log_file_extension}'
        with event_log_path.open('wb') as f:
            f.write(event_log)

        configuration.common.log_path = event_log_path
        configuration.common.test_log_path = None

        event_log, csv_path = read_event_log(configuration.common.log_path, configuration.common.log_ids)

    return configuration, event_log, csv_path


def _is_header_yaml(header: bytes) -> bool:
    return b'application/x-yaml' in header or \
        b'application/yaml' in header or \
        b'text/yaml' in header or \
        b'text/x-yaml' in header or \
        b'text/vnd.yaml' in header


def _infer_event_log_file_extension_from_header(
        request_id: str,
        request_status: RequestStatus,
        header: bytes,
) -> str:
    if b'text/csv' in header:
        return '.csv'
    elif b'application/xml' in header or b'text/xml' in header:
        return '.xml'
    else:
        raise UnsupportedMediaType(
            request_id=request_id,
            status=request_status,
            archive_url=None,
            message='Unsupported event log file type',
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.simod_http_host,
        port=settings.simod_http_port,
        log_level="info",
    )
