import logging
import tarfile
from pathlib import Path

from simod_http.app import Settings, Request


def make_url_for(request_id: str, path: Path, settings: Settings) -> str:
    if settings.simod_http_port == 80:
        port = ''
    else:
        port = f':{settings.simod_http_port}'
    return f'{settings.simod_http_scheme}://{settings.simod_http_host}{port}/discoveries/{request_id}/{path.name}'


class Archiver:
    """
    Compresses a directory of results.
    """

    def __init__(self, settings: Settings, request: Request, results_dir: Path):
        self.settings = settings
        self.request = request
        self.results_dir = results_dir

    def as_tar_gz(self) -> str:
        """
        Compresses the directory into a tar.gz file and returns the URL to fetch it.
        """
        tar_path = self.request.output_dir / f'{self.request.id}.tar.gz'

        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(self.results_dir, arcname=self.results_dir.name)

        logging.debug(f'Archive: {tar_path}')

        return make_url_for(self.request.id, tar_path, self.settings)
