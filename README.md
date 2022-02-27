# Simod

![Build & Test](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/python-app.yml/badge.svg)

Simod combines several process mining techniques to automate the generation and validation of BPS models.  The only input required by the Simod method is an event log in XES, or CSV format. These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Requirements

- **Python 3.8**+
- **PIP 21.2.3**+ (upgrade with `python -m pip install --upgrade pip`)
- For dependencies, please, check `requirements.txt` or `environment.yml`
- For external tools: **Java 1.8**
- **Prosimos** simulator is available at https://github.com/AutomatedProcessImprovement/Prosimos

## Getting Started

Getting the source code:

```shell
$ git clone https://github.com/AutomatedProcessImprovement/Simod.git
$ git submodule update --init --recursive
```

Python environment can be set up using *Anaconda* from `environment.yml` or using the built-in *venv* module from `requirements.txt`.

To install the CLI-tool from the root directory run:

```shell
$ pip install -e .
```

Invoke the tool with either of these:

```shell
$ simod
$ simod optimize --config_path config/optimize_config.yml
$ simod discover --config_path config/discover_without_model_config.yml  # does automatic model discovery from the log
$ simod discover --config_path config/discover_with_model_config.yml     # no need for model discovery
```

The optimizer finds optimal parameters for a model and saves them in `outputs/<id>/PurchasingExample_canon.json`. 

## Testing

### Using a local development environment

We use `pytest` to run tests on the package:

```shell
$ pytest
```

To run only relatively fast tests, execute:

```shell
$ pytest -m "not slow"
```

Coverage:

```shell
$ pytest -m "not slow" --cov=simod
```

### Using Docker

To run the full test suit:

```shell
$ bash test.sh
```

## Docker

- Docker files are available in the `docker` folder.
- Base image for Simod is available at https://hub.docker.com/r/nokal/simod-base and can be downloaded with `docker pull nokal/simod-base:v1.1.4`. The image has a proper Java version for dependencies, Xvfb (for faking X server for the Java dependencies) and Python 3 installed. It doesn't contain Simod itself.

## Benchmarking in HPC with SLURM

```bash
#!/bin/bash

#SBATCH --partition=main'
#SBATCH -J job_name
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=46G
#SBATCH -t 120:00:00

module load any/jdk/1.8.0_265
module load py-xvfbwrapper
module load any/python/3.8.3-conda
conda activate simod

xvfb-run simod optimize --config_path path_to_config
```

## Benchmarking using Docker

The script below assumes a user has a folder with configuration files for each log. It also require a docker container `nokal/simod` available at https://hub.docker.com/r/nokal/simod. 

```bash
#!/usr/bin/env bash

# pass the output directory when executing this script
OUTPUT_DIR=$1

# array of event logs' names
files=("event_log_1" "event_log_2")

BRANCH_NAME="master"

for LOG_NAME in "${files[@]}"; do
	CONFIG_PATH="config_benchmarking/opt_${LOG_NAME}_conf.yml"
	OUTPUT_PATH="tmp/${OUTPUT_DIR}/${LOG_NAME}"
	OUTPUT_LOG_PATH="tmp/${OUTPUT_DIR}/${LOG_NAME}.log"

	echo "* Removing previous container"
	docker rm simod_benchmarking
	
	echo "* Creating output folder"
	mkdir -p $OUTPUT_PATH
	
	echo "* Running Simod"
	docker run --name simod_benchmarking -v /home/ihar/config:/usr/src/simod/config_benchmarking -v /home/ihar/inputs:/usr/src/simod/inputs nokal/simod bash -c "Xvfb :99 &>/dev/null & disown; cd /usr/src/simod; git checkout ${BRANCH_NAME}; git pull && pip install -e .; simod optimize --config_path ${CONFIG_PATH}" &> $OUTPUT_LOG_PATH

	echo "* Copying output"
	docker container cp simod_benchmarking:/usr/src/simod/outputs $OUTPUT_PATH	
done
```

Sample configuration for PurchasingExample.xes:

```yaml
log_path: inputs/PurchasingExample.xes
mining_alg: sm2
exec_mode: optimizer
repetitions: 5
simulator: bimp
sim_metric: tsd
multitasking: true
add_metrics:
- day_hour_emd
- log_mae
- dl
- mae
structure_optimizer:
  max_eval_s: 40
  concurrency:
  - 0.0
  - 1.0
  epsilon:
  - 0.0
  - 1.0
  eta:
  - 0.0
  - 1.0
  gate_management:
  - discovery
  - equiprobable
  or_rep:
  - true
  - false
  and_prior:
  - true
  - false
time_optimizer:
  max_eval_t: 20
  rp_similarity:
  - 0.5
  - 0.9
  res_dtype:
  - dt247
  - lv917
  arr_dtype:
  - dt247
  - lv917
  res_sup_dis:
  - 0.01
  - 0.3
  res_con_dis:
  - 50
  - 85
  arr_support:
  - 0.01
  - 0.1
  arr_confidence:
  - 1
  - 10
```

## Data format
 
The tool assumes the input is composed of a case identifier, an activity label, a resource attribute (indicating which resource performed the activity), 
and two timestamps: the start timestamp and the end timestamp. The resource attribute is required to discover the available resource pools, timetables, 
and the mapping between activities and resource pools, which are a required element in a BPS model. We require both start and end timestamps for each activity instance to compute the processing time of activities, which is also a required element in a simulation model.

## Execution steps

***Event-log loading:*** Under the General tab, the event log must be selected; if the user requires a new event log, it can be loaded in the folder inputs. Remember, the event log must be in XES or CSV format and contain start and complete timestamps. Then It is necessary to define the execution mode between single and optimizer execution.

***Single execution:*** In this execution mode, the user defines the different preprocessing options of the tool manually to generate a simulation model. The next parameters needed to be defined:

 - *Percentile for frequency threshold (eta):* SplitMiner parameter related to the filter over the incoming and outgoing edges. Between
   0.0, and 1.0.    
 - *Parallelism threshold (epsilon):* SplitMiner parameter related to the number of concurrent relations between events to be captured. Between 0.0 and 1.0. 
 - 	*Non-conformance management:* Simod provides three options to deal with the Non-conformances between the event log and the BPMN discovery model. The first option is the   *removal* of the nonconformant traces been the more natural one. The second option is the *replacement* of the non-conformant traces using the conformant most similar ones. The last option is the *repairs* at the event level by creating or deleting an event when necessary.
 - *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is created. The goal of defining this value is to improve the assessment's accuracy, between 1 and 50.

***Optimizer execution:*** In this execution mode, the user defines a search space, and the tool automatically explores the combination looking for the optimal one. The next parameters needed to be defined:

 - *Percentile for frequency threshold range:* SplitMiner parameter related with the filter over the incoming and outgoing edges. Lower and upper bound between 0.0 and 1.0.
 - *Parallelism threshold range:* SplitMiner parameter related to the number of concurrent relations between events to be captured. Lower and upper bound between 0.0 and 1.0.
 - *Max evaluations:* With this value, Simod defines the number of trials in the search space to be explored using a Bayesian hyperparameter optimizer. Between 1 and 50.
 - *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is created. The goal of defining this value is to improve the accuracy of the assessment. Between 1 and 50.

Once all the parameters are settled, It is time to start the execution and wait for the results.
