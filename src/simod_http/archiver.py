import tarfile
from pathlib import Path

from simod_http.app import Settings, Request, settings

logger = settings.logger


class Archiver:
    """
    Compresses a directory of results.
    """

    def __init__(self, settings: Settings, request: Request, results_dir: Path):
        self.settings = settings
        self.request = request
        self.results_dir = results_dir

        self.output_dir = Path(settings.storage_path) / 'results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_url_for(self, path: Path) -> str:
        if self.settings.external_port == 80:
            port = ''
        else:
            port = f':{self.settings.external_port}'
        return f'{self.settings.external_scheme}://{self.settings.external_host}{port}/{path.name}'

    def as_tar_gz(self) -> str:
        """
        Compresses the directory into a tar.gz file and returns the URL to fetch it.
        """
        tar_path = self.output_dir / f'{self.request.id}.tar.gz'

        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(self.results_dir, arcname=self.results_dir.name)

        logger.debug(f'Archive: {tar_path}')

        return self._make_url_for(tar_path)
