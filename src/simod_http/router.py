from typing import Union, Optional

from fastapi import BackgroundTasks, Response, Form, APIRouter
from fastapi.responses import JSONResponse

from simod.configuration import Configuration
from simod.event_log.utilities import read as read_event_log
from simod_http.app import Response as AppResponse, RequestStatus, NotFound, UnsupportedMediaType, NotSupported, app
from simod_http.background_tasks import run_simod_discovery

router = APIRouter()


@router.get('/discoveries/{request_id}/{file_name}')
async def read_discovery_file(request_id: str, file_name: str):
    """
    Get a file from a discovery request.
    """
    request = app.load_request(request_id)

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


@router.get('/discoveries/{request_id}')
async def read_discovery(request_id: str) -> AppResponse:
    """
    Get the status of the request.
    """
    request = app.load_request(request_id)

    return AppResponse(
        request_id=request_id,
        request_status=request.status,
        archive_url=app.make_results_url_for(request),
    )


@router.post('/discoveries')
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
    request = app.new_request_from_params(callback_url, email)

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

    background_tasks.add_task(run_simod_discovery, request, app)

    return response.json_response(status_code=202)


# Helpers

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
