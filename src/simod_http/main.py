import logging
from pathlib import Path
from typing import Callable, Union

import pandas as pd
from fastapi import FastAPI, BackgroundTasks, Request, Response, Body
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from simod.configuration import Configuration
from simod.event_log.utilities import read as read_event_log
from simod_http.app import Response as AppResponse, RequestStatus, settings, Request as AppRequest, Exceptions
from simod_http.background_tasks import run_simod_discovery

app = FastAPI()
logger = settings.logger


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
async def startup():
    logging_handlers = []
    if settings.log_path is not None:
        logging_handlers.append(logging.FileHandler(settings.log_path))

    if settings.logging_level is not None:
        logging.basicConfig(
            level=settings.logging_level.upper(),
            handlers=logging_handlers,
        )

    logger.debug(f'Application settings: {settings}')


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f'HTTP exception: {exc}')
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.get('/discoveries/{request_id}',
         response_model=AppResponse,
         response_model_exclude_unset=True,
         response_model_exclude_none=True)
def read_discoveries(request_id: str) -> AppResponse:
    """
    Get the status of the request.
    """
    return AppResponse(
        request_id=request_id,
        status=RequestStatus.ACCEPTED,  # TODO: determine the actual status
    )


@app.post('/discoveries',
          response_model=AppResponse,
          response_model_exclude_none=True,
          response_model_exclude_unset=True)
async def create_discovery(
        background_tasks: BackgroundTasks,
        files: list[bytes] = Body(),
) -> JSONResponse:
    global logger

    request = AppRequest.empty(settings.storage_path)

    configuration, event_log, event_log_csv_path = parse_files_from_request(files, request.request_dir)

    if event_log is None:
        raise Exceptions.NotFound('No event log file found')

    request.configuration = configuration
    request.event_log = event_log
    request.event_log_csv_path = event_log_csv_path

    response = AppResponse(request_id=request.id, status=RequestStatus.ACCEPTED)

    background_tasks.add_task(run_simod_discovery, request)

    return JSONResponse(status_code=202, content=response.dict())


def parse_files_from_request(
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
            raise Exceptions.BadMultipartRequest('Each part of the multipart request must have a header and a body')

        if is_header_yaml(header):
            configuration = Configuration.from_stream(body)
            continue

        event_log = body
        event_log_file_extension = infer_event_log_file_extension_from_header(header)

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


def is_header_yaml(header: bytes) -> bool:
    return b'application/x-yaml' in header or \
        b'application/yaml' in header or \
        b'text/yaml' in header or \
        b'text/x-yaml' in header or \
        b'text/vnd.yaml' in header


def infer_event_log_file_extension_from_header(header: bytes) -> str:
    if b'text/csv' in header:
        return '.csv'
    elif b'application/xml' in header or b'text/xml' in header:
        return '.xml'
    else:
        raise Exceptions.UnsupportedMediaType(f'Unsupported media type: {header}')
