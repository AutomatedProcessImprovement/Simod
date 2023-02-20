import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from uvicorn.config import LOGGING_CONFIG

from simod_http.app import RequestStatus, Request as AppRequest, Settings, BaseRequestException, NotFound
from simod_http.router import router

debug = os.environ.get('SIMOD_HTTP_DEBUG', 'false').lower() == 'true'

if debug:
    settings = Settings()
else:
    settings = Settings(_env_file='.env.production')

settings.simod_http_storage_path = Path(settings.simod_http_storage_path)

app = FastAPI()
app.include_router(router)


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


@app.get('/{any_str}')
async def root() -> JSONResponse:
    raise NotFound(
        request_id='N/A',
        request_status=RequestStatus.UNKNOWN,
        message='Not found',
    )


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
