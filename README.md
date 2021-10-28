# Simod

![Build & Test](https://github.com/AutomatedProcessImprovement/Simod/actions/workflows/python-app.yml/badge.svg)

Simod combines several process mining techniques to automate the generation and validation of BPS models.  The only input required by the Simod method is an event log in XES, or CSV format. These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Requirements

- **Python 3.8**+
- **PIP 21.2.3**+ (upgrade with `python -m pip install --upgrade pip`)
- For dependencies, please, check `requirements.txt` or `simod.yml`
- For external tools: **Java 1.8**

## Getting Started

Python environment can be set up using *Anaconda* from `simod.yml` or using the built-in *venv* module from `requirements.txt`.

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

## Getting Started with New Replayer

```shell
$ git clone https://github.com/AutomatedProcessImprovement/Simod.git simod
$ cd simod
$ git checkout new_replayer
$ python3 -m venv venv
$ source venv/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install -e .
```

## Testing

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

## Docker

- Docker files are available in the `docker` folder.
- Base image for Simod is available at https://hub.docker.com/r/nokal/simod-base and can be downloaded with `docker pull nokal/simod-base:v1.1.4`. The image has a proper Java version for dependencies, Xvfb (for faking X server for the Java dependencies) and Python 3 installed. It doesn't contain Simod itself.

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

## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
