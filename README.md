# SiMo-Discoverer

SiMo-Discoverer combines several process mining techniques to fully automate the generation and validation of BPS models. 
The only input required by the Simod method is an eventlog in XES, MXML or CSV format.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code you just need to install Anaconda in your system, and create an environment using the *simo.yml* specification provided in the repository.

### Data format
 
The tool assumes the input is composed by a case identifier, an activity label, a resource attribute (indicating which resource performed the activity), 
and two timestamps: the start timestamp and the end timestamp. The resource attribute is required in order to discover the available resource pools, their timetables, 
and the mapping between activities and resource pools, which are a required element in a BPS model. We require both start and endtimestamps for each activity instance, 
in order to compute the processing time of activities, which is also a required element in a simulation model.

### Configuration

Once created the environment you can open the file Simod.ipynb using Jupyter, and execute the first cell of the Notebook. 

## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
