import logging
import shutil
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from uvicorn.config import LOGGING_CONFIG

from simod_http.app import RequestStatus, BaseRequestException, NotFound, app
from simod_http.router import router

api = FastAPI()
api.include_router(router)


@api.on_event('startup')
async def application_startup():
    logging_handlers = []
    if app.simod_http_log_path is not None:
        logging_handlers.append(logging.FileHandler(app.simod_http_log_path, mode='w'))

    if len(logging_handlers) > 0:
        logging.basicConfig(
            level=app.simod_http_logging_level.upper(),
            handlers=logging_handlers,
            format=app.simod_http_logging_format,
        )
    else:
        logging.basicConfig(
            level=app.simod_http_logging_level.upper(),
            format=app.simod_http_logging_format,
        )

    logging.debug(f'Application settings: {app}')


@api.on_event('shutdown')
async def application_shutdown():
    requests_dir = Path(app.simod_http_storage_path) / 'requests'

    if not requests_dir.exists():
        return

    for request_dir in requests_dir.iterdir():
        logging.info(f'Checking request directory before shutting down: {request_dir}')

        await _remove_empty_or_orphaned_request_dir(request_dir)

        try:
            request = app.load_request(request_dir.name)
        except Exception as e:
            logging.error(f'Failed to load request: {request_dir.name}, {str(e)}')
            continue

        # At the end, there are only 'failed' or 'succeeded' requests
        if request.status not in [RequestStatus.SUCCESS, RequestStatus.FAILURE]:
            request.status = RequestStatus.FAILURE
            request.timestamp = pd.Timestamp.now()
            request.save()


@api.on_event('startup')
@repeat_every(seconds=app.simod_http_storage_cleaning_timedelta)
async def clean_up():
    requests_dir = Path(app.simod_http_storage_path) / 'requests'

    if not requests_dir.exists():
        return

    current_timestamp = pd.Timestamp.now()
    expire_after_delta = pd.Timedelta(seconds=app.simod_http_request_expiration_timedelta)

    for request_dir in requests_dir.iterdir():
        if request_dir.is_dir():
            logging.info(f'Checking request directory for expired data: {request_dir}')

            await _remove_empty_or_orphaned_request_dir(request_dir)

            try:
                request = app.load_request(request_dir.name)
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


@api.exception_handler(BaseRequestException)
async def request_exception_handler(_, exc: BaseRequestException) -> JSONResponse:
    logging.error(f'Request exception occurred: {exc}')
    return exc.json_response()


@api.get('/{any_str}')
async def root() -> JSONResponse:
    raise NotFound(
        request_id='N/A',
        request_status=RequestStatus.UNKNOWN,
        message='Not found',
    )


if __name__ == '__main__':
    logging_config = LOGGING_CONFIG
    logging_config['formatters']['default']['fmt'] = app.simod_http_logging_format
    logging_config['formatters']['access']['fmt'] = app.simod_http_logging_format.replace(
        '%(message)s', '%(client_addr)s - "%(request_line)s" %(status_code)s')

    uvicorn.run(
        'main:api',
        host=app.simod_http_host,
        port=app.simod_http_port,
        log_level='info',
        log_config=logging_config,
    )
