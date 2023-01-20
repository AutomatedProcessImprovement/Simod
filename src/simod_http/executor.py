import logging
import tempfile
import traceback
from pathlib import Path
from typing import Union

import pandas as pd

from simod.configuration import Configuration
from simod.event_log.event_log import EventLog
from simod.event_log.preprocessor import Preprocessor
from simod.event_log.utilities import read
from simod.optimization.optimizer import Optimizer
from simod_http.app import Settings, Request, RequestStatus, InternalServerError, NotificationMethod
from simod_http.archiver import Archiver
from simod_http.notifiers import Notifier, EmailNotifier


class Executor:
    """
    Job executor that runs Simod with the user's configuration.
    """

    def __init__(self, app_settings: Settings, request: Request):
        self.settings = app_settings
        self.request = request

    def run(self):
        self.request.status = RequestStatus.RUNNING
        self.request.save()

        with tempfile.TemporaryDirectory() as output_dir:
            logging.debug(f'Simod has been started for the request with id={self.request.id}, output_dir={output_dir}')

            try:
                result_dir = optimize_with_simod(
                    self.request.id,
                    self.request.status,
                    self.request.configuration,
                    self.request.event_log,
                    self.request.event_log_csv_path,
                    Path(output_dir),
                )

                archive_url = Archiver(self.settings, self.request, result_dir).as_tar_gz()
                self.request.archive_url = archive_url
                self.request.status = RequestStatus.SUCCESS
                self.request.timestamp = pd.Timestamp.now()
                self.request.save()

                logging.debug(f'Archive URL: {archive_url}')

                _notify_with_settings(self.settings, self.request)

            except Exception as e:
                self.request.status = RequestStatus.FAILURE
                self.request.timestamp = pd.Timestamp.now()
                self.request.save()

                logging.exception(e)
                traceback.print_exc()

                _notify_with_settings(self.settings, self.request, e)


def optimize_with_simod(
        request_id: str,
        request_status: RequestStatus,
        configuration: Configuration,
        event_log: pd.DataFrame,
        event_log_csv_path: Path,
        output_dir: Path
) -> Path:
    if output_dir is None:
        raise InternalServerError(
            request_id=request_id,
            request_status=request_status,
            archive_url=None,
            message='Output directory is not specified',
        )

    try:
        preprocessor = Preprocessor(event_log, configuration.common.log_ids)
        processed_log = preprocessor.run(
            multitasking=configuration.preprocessing.multitasking,
        )

        test_log = None
        if configuration.common.test_log_path is not None:
            test_log, _ = read(configuration.common.test_log_path, configuration.common.log_ids)

        event_log = EventLog.from_df(
            log=processed_log,
            # would be split into training and validation if test is provided, otherwise into test too
            log_ids=configuration.common.log_ids,
            process_name=configuration.common.log_path.stem,
            test_log=test_log,
            log_path=configuration.common.log_path,
            csv_log_path=event_log_csv_path,
        )

        Optimizer(configuration, event_log=event_log, output_dir=output_dir).run()

    except Exception as e:
        logging.exception(e)
        traceback.print_exc()
        raise InternalServerError(
            request_id=request_id,
            request_status=request_status,
            archive_url=None,
            message=str(e),
        )

    return output_dir


def _notify_with_settings(settings: Settings, request: Request, error: Union[Exception, None] = None):
    if request.notification_settings is None:
        logging.debug('No notification settings provided')
        return

    if request.notification_settings.method == NotificationMethod.HTTP:
        ok = Notifier(
            archive_url=request.archive_url,
        ).callback(request.notification_settings.callback_url, request.status, error=error)

    elif request.notification_settings.method == NotificationMethod.EMAIL:
        ok = EmailNotifier(
            archive_url=request.archive_url,
            smtp_server=settings.simod_http_smtp_server,
            smtp_port=settings.simod_http_smtp_port,
        ).email(request.notification_settings.email, request.status, error=error)

    else:
        logging.debug(f'Unknown notification method: {request.notification_settings.method}')
        return

    request.notified = ok
    request.save()
