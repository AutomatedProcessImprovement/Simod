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
$ docker pull nokal/simod:latest
```

To start a container:

```shell
$ docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash
```

Use the `resources` directory to store event logs and configuration files. The `outputs` directory will contain the results of Simod. 

To start using Simod, you need to activate the Python environment in the container and start `Xvfb`:

```shell
$ cd /usr/src/Simod
$ poetry shell  # opens a shell with the virtual environment
$ Xvfb :99 &>/dev/null & disown  # starts Xvfb (Split Miner requires an X11 server to be available)
$ simod optimize --config_path <path-to-config>
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
$ pytest
```

To run unit tests, execute:

```shell
$ pytest -m "not integration"
```

Coverage:

```shell
$ pytest -m "not integration" --cov=simod
```

## Simod HTTP

Simod HTTP is a web server for Simod. It provides a REST API for Simod and job management. A user submits a request to Simod HTTP by providing a configuration file, an event log, an optionally a BPMN model. Simod HTTP then runs Simod on the provided data and notifies the user when the job is finished because Simod can take a long time to run depending on the size of the event log and the number of optimization trials in the configuration.

Simod HTTP already includes an installed version of Simod in its Docker image. 

To start with the web service, run:

```shell
$ docker run -it -p 8080:80 nokal/simod_http
```

This gives you access to the web service at `http://localhost:8080`. The OpenAPI specification is available at `http://localhost:8080/docs`.

### Example requests

Submitting a job with a configuration and an event log in using a multipart/mixed request:

```shell
$ curl -X POST --location "http://localhost:8080/discoveries" \
    -H "Content-Type: multipart/mixed; boundary="boundary"" \
    -d "--boundary
Content-Disposition: attachment; filename=\"configuration.yaml\"
Content-Type: application/yaml

version: 2
common:
  log_path: /Users/ihar/Projects/PIX/simod/tests/assets/Production_train.csv
  repetitions: 2
  log_ids:
    case: case:concept:name
    start_time: start_timestamp
    end_time: time:timestamp
    activity: concept:name
    resource: org:resource
  evaluation_metrics:
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
preprocessing:
  multitasking: false
structure:
  optimization_metric: dl
  max_evaluations: 2
  mining_algorithm: sm3
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gateway_probabilities:
    - equiprobable
    - discovery
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 2
  resource_profiles:
    discovery_type: differentiated
    granularity:
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01
      - 0.3
    participation: 0.4
extraneous_activity_delays:
  optimization_metric: absolute_emd
  num_iterations: 1

--boundary
Content-Disposition: attachment; filename=\"LoanApp_sequential_9-5_diffres_timers.csv\"
Content-Type: text/csv

case:concept:name,concept:name,start_timestamp,time:timestamp,org:resource
0,A,2022-05-16T10:00:00.000,2022-05-16T10:15:00.000,Marcus
0,B,2022-05-16T12:00:00.000,2022-05-16T12:30:00.000,Anya
0,C,2022-05-16T12:30:00.000,2022-05-16T12:40:00.000,Dom
0,D,2022-05-16T12:40:00.000,2022-05-16T12:55:00.000,Dom
1,A,2022-05-16T10:15:00.000,2022-05-16T10:30:00.000,Marcus
1,B,2022-05-16T13:00:00.000,2022-05-16T13:30:00.000,Anya
1,C,2022-05-16T13:30:00.000,2022-05-16T13:47:00.000,Dom
1,D,2022-05-16T13:47:00.000,2022-05-16T13:59:00.000,Dom
2,A,2022-05-16T10:30:00.000,2022-05-16T10:45:00.000,Marcus
2,B,2022-05-16T12:30:00.000,2022-05-16T13:00:00.000,Anya
2,C,2022-05-16T13:00:00.000,2022-05-16T13:17:00.000,Carmine
2,D,2022-05-16T13:17:00.000,2022-05-16T13:35:00.000,Carmine
3,A,2022-05-16T10:45:00.000,2022-05-16T11:00:00.000,Marcus
3,B,2022-05-16T13:30:00.000,2022-05-16T14:00:00.000,Anya
3,C,2022-05-16T14:00:00.000,2022-05-16T14:21:00.000,Carmine
3,D,2022-05-16T14:31:00.000,2022-05-16T14:42:00.000,Carmine
4,E,2022-05-16T10:27:00.000,2022-05-16T11:30:00.000,Anya
4,F,2022-05-16T11:30:00.000,2022-05-16T12:00:00.000,Anya
4,G,2022-05-16T12:30:00.000,2022-05-16T13:25:00.000,Marcus
5,A,2022-05-17T10:00:00.000,2022-05-17T10:15:00.000,Marcus
5,B,2022-05-17T12:00:00.000,2022-05-17T12:30:00.000,Anya
5,C,2022-05-17T12:30:00.000,2022-05-17T12:40:00.000,Dom
5,D,2022-05-17T12:40:00.000,2022-05-17T12:55:00.000,Dom
6,A,2022-05-17T10:15:00.000,2022-05-17T10:30:00.000,Marcus
6,B,2022-05-17T13:00:00.000,2022-05-17T13:30:00.000,Anya
6,C,2022-05-17T13:30:00.000,2022-05-17T13:47:00.000,Dom
6,D,2022-05-17T13:47:00.000,2022-05-17T13:59:00.000,Dom
7,A,2022-05-17T10:30:00.000,2022-05-17T10:45:00.000,Marcus
7,B,2022-05-17T12:30:00.000,2022-05-17T13:00:00.000,Anya
7,C,2022-05-17T13:00:00.000,2022-05-17T13:17:00.000,Carmine
7,D,2022-05-17T13:17:00.000,2022-05-17T13:35:00.000,Carmine
8,A,2022-05-17T10:45:00.000,2022-05-17T11:00:00.000,Marcus
8,B,2022-05-17T13:30:00.000,2022-05-17T14:00:00.000,Anya
8,C,2022-05-17T14:00:00.000,2022-05-17T14:21:00.000,Carmine
8,D,2022-05-17T14:31:00.000,2022-05-17T14:42:00.000,Carmine
9,E,2022-05-17T10:27:00.000,2022-05-17T11:30:00.000,Anya
9,F,2022-05-17T11:30:00.000,2022-05-17T12:00:00.000,Anya
9,G,2022-05-17T12:30:00.000,2022-05-17T13:25:00.000,Marcus
10,A,2022-05-18T10:00:00.000,2022-05-18T10:15:00.000,Marcus
10,B,2022-05-18T12:00:00.000,2022-05-18T12:30:00.000,Anya
10,C,2022-05-18T12:30:00.000,2022-05-18T12:40:00.000,Dom
10,D,2022-05-18T12:40:00.000,2022-05-18T12:55:00.000,Dom
11,A,2022-05-18T10:15:00.000,2022-05-18T10:30:00.000,Marcus
11,B,2022-05-18T13:00:00.000,2022-05-18T13:30:00.000,Anya
11,C,2022-05-18T13:30:00.000,2022-05-18T13:47:00.000,Dom
11,D,2022-05-18T13:47:00.000,2022-05-18T13:59:00.000,Dom
12,A,2022-05-18T10:30:00.000,2022-05-18T10:45:00.000,Marcus
12,B,2022-05-18T12:30:00.000,2022-05-18T13:00:00.000,Anya
12,C,2022-05-18T13:00:00.000,2022-05-18T13:17:00.000,Carmine
12,D,2022-05-18T13:17:00.000,2022-05-18T13:35:00.000,Carmine
13,A,2022-05-18T10:45:00.000,2022-05-18T11:00:00.000,Marcus
13,B,2022-05-18T13:30:00.000,2022-05-18T14:00:00.000,Anya
13,C,2022-05-18T14:00:00.000,2022-05-18T14:21:00.000,Carmine
13,D,2022-05-18T14:31:00.000,2022-05-18T14:42:00.000,Carmine
14,E,2022-05-18T10:27:00.000,2022-05-18T11:30:00.000,Anya
14,F,2022-05-18T11:30:00.000,2022-05-18T12:00:00.000,Anya
14,G,2022-05-18T12:30:00.000,2022-05-18T13:25:00.000,Marcus"
```

To check the status of the job, you can use the following command:

```shell
$ curl -X GET --location "http://localhost:8080/discoveries/85dee0e3-9614-4c6e-addc-8d126fbc5829"
```

Because a single job can take a long time to complete, you can also provide a callback HTTP endpoint for Simod HTTP to call when the job is ready. The request would like like this:

```shell
$ curl -X POST --location "http://localhost:8080/discoveries?callback_url=http//youdomain.com/callback" \
    -H "Content-Type: multipart/mixed; boundary="boundary"" \
    -d "--boundary
Content-Disposition: attachment; filename=\"configuration.yaml\"
Content-Type: application/yaml

<the rest of the request like it is shown above>
```