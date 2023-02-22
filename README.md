# Simod: Automated discovery of business process simulation models

![Simod](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/simod.yml/badge.svg)
![Simod HTTP](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/simod_http.yml/badge.svg)

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
docker pull nokal/simod:latest
```

To start a container:

```shell
docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash
```

Use the `resources` directory to store event logs and configuration files. The `outputs` directory will contain the results of Simod. 

To start using Simod, you need to activate the Python environment in the container and start `Xvfb`:

```shell
cd /usr/src/Simod
poetry shell  # opens a shell with the virtual environment
Xvfb :99 &>/dev/null & disown  # starts Xvfb (Split Miner requires an X11 server to be available)
simod optimize --config_path <path-to-config>
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
  enable_time_concurrency_threshold: 0.75 # Threshold to consider two activities as concurrent when computing the enabled time.
  concurrency_df: 0.9 # Directly-Follows threshold for the heuristics' concurrency oracle (only used to estimate start times if needed).
  concurrency_l2l: 0.9 # Length 2 loops threshold for the heuristics' concurrency oracle.
  concurrency_l1l: 0.9 # Length 1 loops threshold for the heuristics' concurrency oracle.
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

Simod HTTP is a web server for Simod. It provides a REST API for Simod and job management. A user submits a request to Simod HTTP by providing a configuration file, an event log, an optionally a BPMN model. Simod HTTP then runs Simod on the provided data and notifies the user when the job is finished because Simod can take a long time to run depending on the size of the event log and the number of optimization trials in the configuration.

Simod HTTP already includes an installed version of Simod in its Docker image. 

To start with the web service, run:

```shell
docker run -it -p 8080:80 nokal/simod-http
```

This gives you access to the web service at `http://localhost:8080`. The OpenAPI specification is available at `http://localhost:8080/docs`.

### Example requests

Submitting a job with a configuration and an event log in using a `multipart/form-data` request:

```shell
curl -X POST --location "http://localhost:8080/discoveries" \
-F "configuration=@resources/config/sample.yml; type=application/yaml" \
-F "event_log=@resources/event_logs/PurchasingExample.csv; type=text/csv”
```

`resources/config/sample.yml` is the path to the configuration file and `resources/event_logs/PurchasingExample.csv` is the path to the event log. The type of the files better be specified.

To check the status of the job, you can use the following command:

```shell
curl -X GET --location "http://localhost:8080/discoveries/85dee0e3-9614-4c6e-addc-8d126fbc5829"
```

Because a single job can take a long time to complete, you can also provide a callback HTTP endpoint for Simod HTTP to call when the job is ready. The request would look like this:

```shell
curl -X POST --location "http://localhost:8080/discoveries?callback_url=http//youdomain.com/callback" \
-F "configuration=@resources/config/sample.yml; type=application/yaml" \
-F "event_log=@resources/event_logs/PurchasingExample.csv; type=text/csv”
```