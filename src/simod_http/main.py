import logging
from typing import Callable

from fastapi import FastAPI, BackgroundTasks, Request, Body
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from simod.configuration import Configuration
from simod_http.app import Response, RequestStatus, settings

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


@app.post('/discoveries',
          response_model=Response,
          response_model_exclude_none=True,
          response_model_exclude_unset=True)
async def create_discovery(
        # configuration: Configuration,
        background_tasks: BackgroundTasks,
        files: list[bytes] = Body(),
) -> JSONResponse:
    global logger

    # request = Request.from_configuration(configuration)

    # response = Response(request_id=request.id, status=RequestStatus.ACCEPTED)

    # logger.debug(f'Discovery request accepted: {request}')

    # background_tasks.add_task(test, request)
    # background_tasks.add_task(run_simod_discovery, request)

    print(f'Length of files: {len(files)}')
    print(f'Files: {files}')

    if len(files) != 2:
        return JSONResponse(status_code=400, content={'message': 'Invalid number of files'})

    yaml_content_types = [b'application/x-yaml', b'text/yaml', b'text/x-yaml', b'text/vnd.yaml']

    configuration = None
    event_log = None

    for content in files:
        header, body = content.split(b'\n\n')

        for content_type in yaml_content_types:
            if content_type in header:
                configuration = Configuration.parse_raw(body.decode())
                break

        if configuration is None:
            event_log = body

    if configuration is None:
        return JSONResponse(status_code=400, content={'message': 'No configuration file found'})

    if event_log is None:
        return JSONResponse(status_code=400, content={'message': 'No event log file found'})

    # return JSONResponse(content=response.dict(), status_code=202)
    return JSONResponse(content={}, status_code=202)
