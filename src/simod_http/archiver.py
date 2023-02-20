import logging
import tarfile
from pathlib import Path

from simod_http.app import Application, Request


class Archiver:
    """
    Compresses a directory of results.
    """

    def __init__(self, app: Application, request: Request, results_dir: Path):
        self.app = app
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

        return self.app.make_results_url_for(self.request)
