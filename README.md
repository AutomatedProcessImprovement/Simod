# Simod: Automated discovery of business process simulation models

![Simod](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/simod.yml/badge.svg)

Simod combines several process mining techniques to automate the generation and validation of BPS models. The only input required by the Simod method is an event log in XES, or CSV format, and a configuration file.

This repository contains 2 projects:

- `src/simod` - the main Simod package
- `src/simod_http` - a web server for Simod

## Simod

### Requirements

- Python 3.9
- Java 1.8 is required by Split Miner which is used for process model discovery
- Use [Poetry](https://python-poetry.org/) for building, installing, and managing Python dependencies

### Getting Started

```shell
docker pull nokal/simod:v3.2.1
```

To start a container:

```shell
docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod:v3.2.1 bash
```

Use the `resources` directory to store event logs and configuration files. The `outputs` directory will contain the results of Simod. 

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

### Configuration

Example configuration with description and possible values:

```yaml
version: 2
common:
  log_path: resources/event_logs/PurchasingExample.xes  # Path to the event log in XES or CSV format
  test_log_path: resources/event_logs/PurchasingExampleTest.xes  # Optional: Path to the test event log in XES or CSV format
  repetitions: 1  # Number of times that the evaluation of each candidate is run (included the final model) during the optimization. The evaluation metric of the candidate is the average of it's repetitions evaluations.
  evaluation_metrics: # A list of evaluation metrics to use on the final model
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
preprocessing: # Event log preprocessing settings
  multitasking: false # If true, remove the multitasking by adjusting the timestamps (start/end) of those activities being executed at the same time by the same resource.
  concurrency_df: 0.9 # Directly-Follows threshold for the concurrency oracle.
  concurrency_l2l: 0.9 # Length 2 loops threshold for the concurrency oracle.
  concurrency_l1l: 0.9 # Length 1 loops threshold for the concurrency oracle.
structure: # Structure settings
  optimization_metric: dl  # Optimization metric for the structure. DL or N_GRAM_DISTANCE
  max_evaluations: 1  # Number of optimization iterations over the search space. Values between 1 and 50
  mining_algorithm: sm3  # Process model discovery algorithm. Options: sm1, sm2, sm3 (recommended)
  concurrency: # Split Miner 2 (sm2) parameter for the number of concurrent relations between events to be captured. Values between 0.0 and 1.0
    - 0.0
    - 1.0
  epsilon: # Split Miner 1 and 3 (sm1, sm3) parameter specifying the number of concurrent relations between events to be captured. Values between 0.0 and 1.0
    - 0.0
    - 1.0
  eta: # Split Miner 1 and 3 (sm1, sm3) parameter specifying the filter over the incoming and outgoing edges. Values between 0.0 and 1.0
    - 0.0
    - 1.0
  gateway_probabilities: # Methods of discovering gateway probabilities. Options: equiprobable, discovery
    - equiprobable
    - discovery
  replace_or_joins: # Split Miner 3 (sm3) parameter specifying whether to replace non-trivial OR joins or not. Options: true, false
    - true
    - false
  prioritize_parallelism: # Split Miner 3 (sm3) parameter specifying whether to prioritize parallelism over loops or not. Options: true, false
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd  # Optimization metric for the calendars. Options: absolute_hourly_emd, cycle_time_emd, circadian_emd
  max_evaluations: 1  # Number of optimization iterations over the search space. Values between 1 and 50
  resource_profiles: # Resource profiles settings
    discovery_type: pool  # Resource profile discovery type. Options: differentiated, pool, undifferentiated
    granularity: # Time granularity for calendars in minutes. Bigger logs will benefit from smaller granularity
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01
      - 0.3
    participation: 0.4  # Resource participation threshold. Values between 0.0 and 1.0
extraneous_activity_delays: # Settings for extraneous activity timers discovery
  num_iterations: 1  # Number of optimization iterations over the search space. Values between 1 and 50
  optimization_metric: relative_emd # Optimization metric for the extraneous activity timers. Options: relative_emd, absolute_emd, circadian_emd, cycle_time
```

**NB!** Split Miner 1 is not supported anymore. Split Miner 3 will be executed instead.

### Testing

Use `pytest` to run tests on the package:

```shell
pytest
```

To run unit tests, execute:

```shell
pytest -m "not integration"
```

Coverage:

```shell
pytest -m "not integration" --cov=simod
```

## Simod HTTP

Since Simod v3.3.0 its subproject Simod HTTP has moved as one of the services to https://github.com/AutomatedProcessImprovement/simod-on-containers.