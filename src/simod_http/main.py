import logging
import os
import shutil
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, Response, Form
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from uvicorn.config import LOGGING_CONFIG

from simod.configuration import Configuration
from simod.event_log.utilities import read as read_event_log
from simod_http.app import Response as AppResponse, RequestStatus, Request as AppRequest, Settings, NotFound, \
    UnsupportedMediaType, BaseRequestException, NotificationSettings, NotificationMethod, \
    NotSupported
from simod_http.archiver import make_url_for
from simod_http.executor import Executor

debug = os.environ.get('SIMOD_HTTP_DEBUG', 'false').lower() == 'true'

if debug:
    settings = Settings()
else:
    settings = Settings(_env_file='.env.production')

settings.simod_http_storage_path = Path(settings.simod_http_storage_path)

app = FastAPI()


# Background tasks

def run_simod_discovery(request: Request, settings: Settings):
    """
    Run Simod with the user's configuration.
    """
    executor = Executor(app_settings=settings, request=request)
    executor.run()


# Hooks

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

    if not requests_dir.exists():
        return

    for request_dir in requests_dir.iterdir():
        logging.info(f'Checking request directory before shutting down: {request_dir}')

        await _remove_empty_or_orphaned_request_dir(request_dir)

        try:
            request = AppRequest.load(request_dir.name, settings)
        except Exception as e:
            logging.error(f'Failed to load request: {request_dir.name}, {str(e)}')
            continue

        # At the end, there are only 'failed' or 'succeeded' requests
        if request.status not in [RequestStatus.SUCCESS, RequestStatus.FAILURE]:
            request.status = RequestStatus.FAILURE
            request.timestamp = pd.Timestamp.now()
            request.save()


@app.on_event('startup')
@repeat_every(seconds=settings.simod_http_storage_cleaning_timedelta)
async def clean_up():
    requests_dir = Path(settings.simod_http_storage_path) / 'requests'

    if not requests_dir.exists():
        return

    current_timestamp = pd.Timestamp.now()
    expire_after_delta = pd.Timedelta(seconds=settings.simod_http_request_expiration_timedelta)

    for request_dir in requests_dir.iterdir():
        if request_dir.is_dir():
            logging.info(f'Checking request directory for expired data: {request_dir}')

            await _remove_empty_or_orphaned_request_dir(request_dir)

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


async def _remove_empty_or_orphaned_request_dir(request_dir):
    # Removes empty directories
    if len(list(request_dir.iterdir())) == 0:
        logging.info(f'Removing empty directory: {request_dir}')
        shutil.rmtree(request_dir, ignore_errors=True)

    # Removes orphaned request directories
    if not (request_dir / 'request.json').exists():
        logging.info(f'Removing request folder for {request_dir.name}, no request.json file')
        shutil.rmtree(request_dir, ignore_errors=True)


@app.exception_handler(BaseRequestException)
async def request_exception_handler(_, exc: BaseRequestException) -> JSONResponse:
    logging.error(f'Request exception occurred: {exc}')
    return exc.json_response()


# Routes

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
        archive_url=make_url_for(request.id, Path(f'{request.id}.tar.gz'),
                                 settings) if request.status == RequestStatus.SUCCESS else None,
    )


@app.post('/discoveries')
async def create_discovery(
        background_tasks: BackgroundTasks,
        configuration=Form(),
        event_log=Form(),
        callback_url: Optional[str] = None,
        email: Optional[str] = None,
) -> JSONResponse:
    """
    Create a new business process model discovery and optimization request.
    """
    global settings

    request = await _empty_request_from_params(settings.simod_http_storage_path, callback_url, email)

    if email is not None:
        request.status = RequestStatus.FAILURE
        request.save()

        raise NotSupported(
            request_id=request.id,
            request_status=request.status,
            message='Email notifications are not supported',
        )

    # Configuration

    configuration = Configuration.from_stream(configuration.file)

    # Event log

    event_log_file_extension = _infer_event_log_file_extension_from_header(event_log.content_type)
    if event_log_file_extension is None:
        raise UnsupportedMediaType(
            request_id=request.id,
            request_status=request.status,
            archive_url=None,
            message='Unsupported event log file type',
        )

    event_log_path = request.output_dir / f'event_log{event_log_file_extension}'
    event_log_path.write_bytes(event_log.file.read())

    configuration.common.log_path = event_log_path.absolute()
    configuration.common.test_log_path = None

    event_log, event_log_csv_path = read_event_log(configuration.common.log_path, configuration.common.log_ids)

    request.configuration = configuration
    request.event_log = event_log
    request.event_log_csv_path = event_log_csv_path
    request.status = RequestStatus.ACCEPTED
    request.save()

    # Response

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


# Helpers

async def _empty_request_from_params(
        storage_path: str,
        callback_url: Optional[str] = None,
        email: Optional[str] = None
) -> AppRequest:
    request = AppRequest.empty(Path(storage_path))

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


def _infer_event_log_file_extension_from_header(
        content_type: str,
) -> Union[str, None]:
    if 'text/csv' in content_type:
        return '.csv'
    elif 'application/xml' in content_type or 'text/xml' in content_type:
        return '.xml'
    else:
        return None


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
