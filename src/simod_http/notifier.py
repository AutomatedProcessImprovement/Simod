from typing import Union


class Notifier:
    """
    Notifier informs the user about the results of the Simod discovery.
    """

    def __init__(self, archive_url: Union[str, None] = None):
        self.archive_url = archive_url

    def callback(self, callback_url: str, error: Union[Exception, None] = None):
        """
        Sends a callback request to the user's server.
        """
        print(f'Callback URL: {callback_url}')
