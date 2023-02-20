from fastapi import Request

from simod_http.app import Application
from simod_http.executor import Executor


def run_simod_discovery(request: Request, settings: Application):
    """
    Run Simod with the user's configuration.
    """
    executor = Executor(app_settings=settings, request=request)
    executor.run()
