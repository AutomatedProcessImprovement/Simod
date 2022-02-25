from setuptools import setup, find_packages

setup(
    name='simod',
    version='3.1.0',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'click',
        'pandas',
        'numpy',
        'networkx',
        'matplotlib',
        'lxml',
        'xmltodict',
        'jellyfish',
        'scipy',
        'tqdm',
        'PyYAML',
        'hyperopt',
        'pytz',
        'pytest',
        'setuptools',
    ],
    entry_points={
        'console_scripts': [
            'simod = simod.cli:main',
        ]
    }
)
