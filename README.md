# Simod: Automated discovery of business process simulation models

![Simod](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/simod.yml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/simod)

Simod combines process mining and machine learning techniques to automate the discovery and tuning of Business Process Simulation models from event logs extracted from enterprise information systems (ERPs, CRM, case management systems, etc.).
Simod takes as input an event log in CSV format, a configuration file, and (optionally) a BPMN process model, and returns a business process simulation scenario that can be simulated using the [Prosimos](https://github.com/AutomatedProcessImprovement/Prosimos) simulator, which is embedded in Simod.

## Requirements

- Python 3.9
- Java 1.8 is required by Split Miner which is used for process model discovery
- Use [Poetry](https://python-poetry.org/) for building, installing, and managing Python dependencies

## Getting Started

```shell
docker pull nokal/simod:v3.2.1
```

To start a container:

```shell
docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod:v3.2.1 bash
```

Use the `resources` directory to store event logs and configuration files. The `outputs` directory will contain the
results of Simod.

To start using Simod, you need to activate the Python environment in the container and start `Xvfb`:

```shell
cd /usr/src/Simod
Xvfb :99 &>/dev/null & disown  # starts Xvfb (Split Miner requires an X11 server to be available)
poetry run simod optimize --config_path <path-to-config>
```

Starting from v3.2.1, the above command can be simplified to:

```shell
bash run.sh <path-to-config> <optional-path-to-output-dir>
```

Different Simod versions are available at https://hub.docker.com/r/nokal/simod/tags.

## Configuration file

A set of example configurations can be found in the 
[resources](https://github.com/AutomatedProcessImprovement/Simod/tree/master/resources) folder along with a description
of each element:
- Basic configuration to discover the full BPS model ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example.yml)).
- Basic configuration to discover the full BPS model, and evaluate it with a specified event log ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example_with_evaluation.yml)).
- Basic configuration to discover a BPS model with a provided BPMN process model as starting point ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/configuration_example_with_provided_process_model.yml)).
- Complete configuration example with all the possible parameters ([here](https://github.com/AutomatedProcessImprovement/Simod/blob/master/resources/config/complete_configuration.yml)).

You can run any of these examples by executing the following command:

```shell
poetry run simod optimize --config_path resources/config/configuration_example.yml
```

## Testing

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
