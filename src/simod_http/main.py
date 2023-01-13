import logging

from fastapi import FastAPI, BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from simod.configuration import Configuration
from simod_http.app import Response, Request, RequestStatus, settings
from simod_http.background_tasks import run_simod_discovery, test

app = FastAPI()
logger = settings.logger


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
         response_model=Response,
         response_model_exclude_unset=True,
         response_model_exclude_none=True)
def read_discoveries(request_id: str):
    """
    Get the status of the request.
    """
    return Response(
        request_id=request_id,
        status=RequestStatus.ACCEPTED,  # TODO: determine the actual status
    )


@app.post('/discoveries', response_model=Response, response_model_exclude_none=True, response_model_exclude_unset=True)
async def create_discovery(configuration: Configuration, background_tasks: BackgroundTasks) -> JSONResponse:
    global logger

    request = Request.from_configuration(configuration)

    response = Response(request_id=request.id, status=RequestStatus.ACCEPTED)

    logger.debug(f'Discovery request accepted: {request}')

    # background_tasks.add_task(test, request)
    background_tasks.add_task(run_simod_discovery, request)

    return JSONResponse(content=response.dict(), status_code=202)
