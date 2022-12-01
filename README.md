# Simod

![CI](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/ci.yml/badge.svg)

Simod combines several process mining techniques to automate the generation and validation of BPS models. The only input
required by the Simod method is an event log in XES, or CSV format. These instructions will get you a copy of the
project up and running on your local machine for development and testing purposes.

## Requirements

- **Python 3.8**+
- **PIP 21.2.3**+ (upgrade with `python -m pip install --upgrade pip`)
- For Python dependencies, see `requirements.txt`
- For external tools: **Java 1.8**
- Java dependencies alongside with others are located at `external_tools` folder

## Installation via Docker

```shell
$ docker pull nokal/simod:latest
```

To start a container:

```shell
$ docker run -it nokal/simod:latest bash
```

In the container, you need to activate the Python environment pre-installed during Docker building:

```shell
> cd /usr/src/Simod
> source venv/bin/activate
> simod --help
```

Base image for Simod is available at https://hub.docker.com/r/nokal/simod-base and can be downloaded
with `docker pull nokal/simod-base:v1.1.6`. The image has a proper Java version for dependencies, Xvfb (for faking X
server for the Java dependencies) and Python 3 installed. It doesn't contain Simod itself.

Different Simod versions are available at https://hub.docker.com/r/nokal/simod/tags.

## Installation from source

Getting the source:

```shell
$ git clone https://github.com/AutomatedProcessImprovement/Simod.git Simod
$ cd Simod
$ git checkout master
$ git submodule update --init --recursive
```

### PIP

Creating the virtual environment:

```shell
$ python3 -m venv venv
$ source $VENV_DIR/bin/activate
$ pip install --upgrade pip
```

Installing the dependencies:

```shell
$ cd external_tools/Prosimos
$ pip install -e .
$ cd ../pm4py-wrapper
$ pip install -e .
```

Installing Simod:

```shell
$ cd ../..
$ pip install -e .
```

## Getting started

```shell
$ simod optimize --config_path <path-to-config>
```

The optimizer finds optimal parameters for a model and saves them in `outputs/<id>/<event-log-name>_canon.json`.

## Configuration

Example configuration with description and possible values:

```yaml
version: 2
common:
  log_path: resources/event_logs/PurchasingExample.xes  # Path to the event log in XES or CSV format
  test_log_path: resources/event_logs/PurchasingExampleTest.xes  # Optional: Path to the test event log in XES or CSV format
  exec_mode: optimizer  # Execution mode: optimizer or single (not used)
  repetitions: 1  # Number of simulations of the final model to obtain more accurate evaluations. Values between 1 and 50
  simulation: true  # (not used)
  evaluation_metrics: # A list of evaluation metrics to use on the final model
    - dl
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
preprocessing: # Event log preprocessing settings
  multitasking: false
structure: # Structure settings
  optimization_metric: dl  # Optimization metric for the structure. Only DL is supported
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
```

## Testing

### Using a local development environment

We use `pytest` to run tests on the package:

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

## Data format

The tool assumes the input is composed of a case identifier, an activity label, a resource attribute (indicating which
resource performed the activity),
and two timestamps: the start timestamp and the end timestamp. The resource attribute is required to discover the
available resource pools, timetables,
and the mapping between activities and resource pools, which are a required element in a BPS model. We require both
start and end timestamps for each activity instance to compute the processing time of activities, which is also a
required element in a simulation model.

## Execution steps

***Event-log loading:*** Under the General tab, the event log must be selected; if the user requires a new event log, it
can be loaded in the folder inputs. Remember, the event log must be in XES or CSV format and contain start and complete
timestamps. Then It is necessary to define the execution mode between single and optimizer execution.

***Single execution:*** In this execution mode, the user defines the different preprocessing options of the tool
manually to generate a simulation model. The next parameters needed to be defined:

- *Percentile for frequency threshold (eta):* SplitMiner parameter related to the filter over the incoming and outgoing
  edges. Between
  0.0, and 1.0.
- *Parallelism threshold (epsilon):* SplitMiner parameter related to the number of concurrent relations between events
  to be captured. Between 0.0 and 1.0.
- *Non-conformance management:* Simod provides three options to deal with the Non-conformances between the event log and
  the BPMN discovery model. The first option is the   *removal* of the nonconformant traces been the more natural one.
  The second option is the *replacement* of the non-conformant traces using the conformant most similar ones. The last
  option is the *repairs* at the event level by creating or deleting an event when necessary.
- *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is
  created. The goal of defining this value is to improve the assessment's accuracy, between 1 and 50.

***Optimizer execution:*** In this execution mode, the user defines a search space, and the tool automatically explores
the combination looking for the optimal one. The next parameters needed to be defined:

- *Percentile for frequency threshold range:* SplitMiner parameter related with the filter over the incoming and
  outgoing edges. Lower and upper bound between 0.0 and 1.0.
- *Parallelism threshold range:* SplitMiner parameter related to the number of concurrent relations between events to be
  captured. Lower and upper bound between 0.0 and 1.0.
- *Max evaluations:* With this value, Simod defines the number of trials in the search space to be explored using a
  Bayesian hyperparameter optimizer. Between 1 and 50.
- *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is
  created. The goal of defining this value is to improve the accuracy of the assessment. Between 1 and 50.

Once all the parameters are settled, It is time to start the execution and wait for the results.
