import shutil
from pathlib import Path
from typing import Optional

from invoke import task, run

azul_jdk = {
    '8': {
        'darwin:arm64': 'https://cdn.azul.com/zulu/bin/zulu8.60.0.21-ca-jdk8.0.322-macosx_aarch64.zip',
        'darwin:amd64': 'https://cdn.azul.com/zulu/bin/zulu8.60.0.21-ca-jdk8.0.322-macosx_x64.zip',
        'linux:amd64': 'https://cdn.azul.com/zulu/bin/zulu8.60.0.21-ca-jdk8.0.322-linux_x64.tar.gz',
        'linux:arm64': 'https://cdn.azul.com/zulu-embedded/bin/zulu8.60.0.21-ca-jdk8.0.322-linux_aarch64.tar.gz',
        'windows:amd64': 'https://cdn.azul.com/zulu/bin/zulu8.60.0.21-ca-jdk8.0.322-win_x64.zip'
    }
}

DARWIN = 'darwin'
LINUX = 'linux'
WINDOWS = 'windows'

ARCH_AMD64 = 'amd64'
ARCH_ARM64 = 'arm64'

vendor_dir = Path('vendor')
java_dir = vendor_dir / 'java'
java_home: Optional[Path] = None


@task
def create_directories(ctx):
    """
    Create the vendor directory.
    """
    java_dir.mkdir(parents=True, exist_ok=True)


@task
def set_home(ctx):
    """
    Set the JAVA_HOME and PATH environment variables.
    """
    run(f'export JAVA_HOME={str(java_dir)}')
    run('export PATH=$JAVA_HOME/bin:$PATH')


@task
def determine_home(ctx):
    """
    Determine the JAVA_HOME environment variable.
    """
    global java_dir, java_home

    if java_home is not None:
        return
    try:
        java_home = Path(list(java_dir.glob('zulu8*/*.jdk/Contents/Home'))[0])
        print(f'● Java home found: {java_home}')
    except IndexError:
        return


@task(pre=[determine_home])
def make_bin_executable(ctx):
    """
    Make the Java bin directory executable.
    """
    _make_files_executable(java_home / 'bin')


@task(pre=[determine_home])
def test_version(ctx, java_version='8'):
    """
    Test the Java version.
    """
    global java_home

    if java_version == '8':
        java_full_version = '1.8.0_322'
    else:
        raise NotImplementedError(f'Unsupported Java version: {java_version}')

    java_bin = str(java_home / 'bin/java')
    result = run(f'{java_bin} -version', hide=True)

    assert java_full_version in result.stdout or java_full_version in result.stderr
    print(f'● Java version {java_full_version} is installed correctly')


@task(pre=[create_directories], post=[set_home, make_bin_executable, test_version])
def install(ctx, platform, arch, java_version='8'):
    """
    Download and extract the Java JDK.
    """
    global java_home, java_dir

    try:
        url = azul_jdk[java_version][f'{platform}:{arch}']
    except KeyError:
        raise NotImplementedError(
            f'Unsupported java version, platform or architecture: {java_version}, {platform}, {arch}')

    file_name = url.split('/')[-1]
    base_dir = Path.cwd()
    downloaded_path = base_dir / file_name
    if not downloaded_path.exists():
        _download_url(url, str(downloaded_path))

    java_home = java_dir / file_name
    if not java_home.exists():
        _extract_archive(downloaded_path, java_dir)


def _download_url(url, output_path):
    """
    Download a file from a URL using cURL.
    """
    print(f'● Downloading to: {output_path}')
    run(f'curl -o {output_path} {url}')


def _extract_archive(input_path, output_dir):
    print(f'● Extracting to: {output_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(input_path, output_dir)


def _make_files_executable(path: Path):
    for file in path.iterdir():
        if file.is_file():
            file.chmod(0o755)
