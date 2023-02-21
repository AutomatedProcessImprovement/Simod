import logging
import smtplib
from typing import Union

import requests

from simod_http.app import Error, RequestStatus


class Notifier:
    """
    Notifier informs the user about the results of the Simod discovery via a callback request.
    """

    def __init__(self, archive_url: Union[str, None] = None):
        self.archive_url = archive_url

    def callback(self, callback_url: str, request_status: RequestStatus, error: Union[Exception, None] = None) -> bool:
        """
        Sends a callback request to the user's server.
        """
        payload = {
            'error': Error(message=str(error)) if error else None,
            'request_status': request_status,
            'archive_url': self.archive_url,
        }
        try:
            response = requests.post(callback_url, json=payload)

            if response.ok:
                logging.info(f'Callback request to {callback_url} was successful, response: {response.json()}')
            else:
                logging.error(f'Callback request to {callback_url} failed, '
                              f'status_code: {response.status_code}, '
                              f'response: {response.json()}')

        except Exception as e:
            logging.error(f'Callback request to {callback_url} failed, error: {e}')
            return False

        return True


class EmailNotifier:
    """
    Notifier informs the user about the results of the Simod discovery via email.
    """

    def __init__(
            self,
            archive_url: Union[str, None] = None,
            smtp_server: str = 'localhost',
            smtp_port: int = 25,
    ):
        self.archive_url = archive_url
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def email(self, email: str, request_status: RequestStatus, error: Union[Exception, None] = None) -> bool:
        """
        Sends an email to the user.
        """
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                error_message = f'Error: {error}\n' if error else ''
                from_addr = 'http@simod.pix.cloud.ut.ee'
                server.sendmail(
                    from_addr=from_addr,
                    to_addrs=email,
                    msg=f'From: {from_addr}\r\n'
                        f'To: {email}\r\n\r\n'
                        f'Subject: Simod discovery request: {request_status}\r\n\r\n'
                        f'Your Simod discovery request has finished with status: {request_status}.\n'
                        f'Archive URL: {self.archive_url}\n'
                        + error_message
                )
        except Exception as e:
            logging.error(f'Email to {email} failed, error: {e}')
            return False

        return True
