# Simod

Simod combines several process mining techniques to fully automate the generation and validation of BPS models.  The only input required by the Simod method is an eventlog in XES, MXML or CSV format. These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *simo.yml* specification provided in the repository.

### Data format
 
The tool assumes the input is composed by a case identifier, an activity label, a resource attribute (indicating which resource performed the activity), 
and two timestamps: the start timestamp and the end timestamp. The resource attribute is required in order to discover the available resource pools, their timetables, 
and the mapping between activities and resource pools, which are a required element in a BPS model. We require both start and endtimestamps for each activity instance, 
in order to compute the processing time of activities, which is also a required element in a simulation model.

### Configuration

Once created the environment you can open the file Simod.ipynb using Jupyter, and execute the first cell of the Notebook.

### Execution steps

***Event-log loading:*** Under the General tab the event log must be selected, if the user requires a new event log it can be loaded in the folder inputs. Remember the event log must be in XES or MXML format and contain start and complete timestamps. Then It is necessary to define the execution mode between single execution, and Optimizer execution.

***Single execution:*** In this execution mode the user defines manually the different preprocessing options of the tool to generate a simulation model. The next parameters needed to be defined:

 - *Percentile for frequency threshold (eta):* SplitMiner parameter
   related with the filter over the incoming and outgoing edges. Between
   0.0 and 1.0.    
 - *Parallelism threshold (epsilon):* SplitMiner parameter related with the quantity of concurrent relations between events to be captured. Between 0.0 and 1.0. 
 - 	*Non-conformance management:* Simod provides three options to deal with the Non-conformances between the eventlog and the BPMN discovery model. The first option is the   *removal* of the nonconformant traces been the more natural one. The second option is the *replacement* of the non-conformant traces using the conformant most similar ones. The last option is the *reparison* at event level by the creation or deletion of an event when it is necessary.
 - *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is created. The goal of define this value is to improve the accuracy of the assessment. Between 1 and 50.

***Optimizer execution:*** In this execution mode the user defines a search space and the tool automatically explore the combination looking for the optimal one. The next parameters needed to be defined:

 - *Percentile for frequency threshold range:* SplitMiner parameter related with the filter over the incoming and outgoing edges. Lower and upper bound between 0.0 and 1.0.
 - *Parallelism threshold range:* SplitMiner parameter related with the quantity of concurrent relations between events to be captured. Lower and upper bound between 0.0 and 1.0.
 - *Max evaluations:* With this values Simod defines the number of trials in the search space to be explored using a Bayesian hyperparameter optimizer. Between 1 and 50.
 - *Number of simulations runs:* Refers to the number of simulations performed by the BIMP simulator, once the model is created. The goal of define this value is to improve the accuracy of the assessment. Between 1 and 50.

Once all the parameters are settled It is time to start the execution and wait for the results.

## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
