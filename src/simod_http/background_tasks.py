from simod_http.app import Request
from simod_http.executor import Executor
from simod_http.main import settings


def run_simod_discovery(request: Request):
    """
    Run Simod with the user's configuration.
    """
    executor = Executor(app_settings=settings, request=request)
    executor.run()


def test(request: Request):
    print(f'Request: {request}, settings: {settings}')
