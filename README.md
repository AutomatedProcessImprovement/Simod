# Simod: Automated discovery of business process simulation models

![Simod](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/simod.yml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/simod)

Simod combines process mining and machine learning techniques to automate the discovery and tuning of Business Process
Simulation models from event logs extracted from enterprise information systems (ERPs, CRM, case management systems,
etc.).
Simod takes as input an event log in CSV format, a configuration file, and (optionally) a BPMN process model, and
returns a business process simulation scenario that can be simulated using
the [Prosimos](https://github.com/AutomatedProcessImprovement/Prosimos) simulator, which is embedded in Simod.

## Dependencies

### Required

| Dependency | Version | Notes                                                                                                                                          |
|------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Python     | 3.9     | For Windows, [Python 3.9.13](https://www.python.org/downloads/release/python-3913/) is the last distribution with Windows installers.          |
| Java       | 1.8     | For example, use [Amazon Corretto 8](https://docs.aws.amazon.com/corretto/latest/corretto-8-ug/downloads-list.html).                           |
| Poetry     | latest  | If using Docker or compiling from source, use [Poetry](https://python-poetry.org/) for building, installing, and managing Python dependencies. |

### Optional

Depending on your CPU architecture, some dependencies might not be pre-compiled for your platform. In that case, you
will
most likely also need the following dependencies:

| Dependency     | Version | Notes                                            |
|----------------|---------|--------------------------------------------------|
| Cargo and Rust | latest  | Install it with [rustup.rs](https://rustup.rs/). |

## Getting Started

### PyPI

❗️Make sure `java -version` returns `1.8` and `pip` is installed.

Then, install Simod and run it with the following commands:

```shell
pip install simod
simod --configuration resources/config/configuration_example.yml
```

Use your own configuration file instead of `resources/config/configuration_example.yml` and specify the path to the 
event log in the configuration file itself. Paths are relative to the configuration file, or absolute.

PyPI project is available at https://pypi.org/project/simod/.

### Docker

```shell
docker pull nokal/simod
```

To start a container:

```shell
docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash
```

Use the `resources` directory to store event logs and configuration files. The `outputs` directory will contain the
results of Simod.

From inside the container, you can run Simod with:

```shell
poetry run simod --configuration <path-to-config>
```

Docker images for different Simod versions are available at https://hub.docker.com/r/nokal/simod/tags.

## Configuration file

A set of example configurations can be found in the
[resources](https://github.com/AutomatedProcessImprovement/Simod/tree/master/resources) folder along with a description
of each element:

- Basic configuration to discover the full BPS
  model ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example.yml)).
- Basic configuration to discover the full BPS model, and evaluate it with a specified event
  log ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example_with_evaluation.yml)).
- Basic configuration to discover a BPS model with a provided BPMN process model as starting
  point ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example_with_provided_process_model.yml)).
- Basic configuration to discover a BPS model with no optimization process (one-shot) ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_one_shot.yml)).
- Complete configuration example with all the possible
  parameters ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/complete_configuration.yml)).

## For developers

### Testing

Use `pytest` to run tests on the package:

```shell
poetry run pytest
```

To run unit tests, execute:

```shell
poetry run pytest -m "not integration"
```

Coverage:

```shell
poetry run pytest -m "not integration" --cov=simod
```
