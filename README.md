# SiMo-Discoverer

SiMo-Discoverer combines several process mining techniques to fully automate the generation and validation of BPS models. 
The only input required by the SiMo-Discoverer method is an eventlog in XES, MXML or CSV format.

### Prerequisites

Java v8
Python 3.6.6 and the following libraries are nedded to execute SiMo discoverer:

* lxml==4.2.5
* matplotlib==2.2.3
* networkx==2.2
* numpy==1.15.4
* seaborn==0.9.0

### Data format
 
The tool assumes the input is composed by a case identifier, an activity label, a resource attribute (indicating which resource performed the activity), 
and two timestamps: the start timestamp and the end timestamp. The resource attribute is required in order to discover the available resource pools, their timetables, 
and the mapping between activities and resource pools, which are a required element in a BPS model. We require both start and endtimestamps for each activity instance, 
in order to compute the processing time of activities, which is also a required element in a simulation model.

### Configuration

The initial hyper-parameters configuration of simo must be carried on the file simo\config.ini. 
Next you can find a short description about the main hyper-parameters of the tool.

* The tag [FILE] - name is related with the eventlog name that will be used asn an input for the BPS model generation
* The tag [EXECUTION] - starttimeformat is related with the string format of the eventlog start timestamp (default %Y-%m-%dT%H:%M:%S.000)
* The tag [EXECUTION] - endtimeformat is related with the string format of the eventlog complete timestamp (default %Y-%m-%dT%H:%M:%S.000)
* * The tag [EXECUTION] - mining enables the generation of BPMN models using the SplitMiner tool
* The tag [EXECUTION] - alignment enables the eventlog trace alignment and repairing using the ProConformancev2 tool
* The tag [EXECUTION] - parameters enables the BPS model extraction
* The tag [EXECUTION] - simulation enables the simulation of a BPS model, using the BIMP or the Scylla simulators (the simulator choise es configured with the tag [SIMULATOR] - simulator)
* The tag [EXECUTION] - analysis enables the generation of a comparison report between the simulation and the direct measurments of the eventlog
* The tag [EXECUTION] - simcycles indicates the desired simulation cycles (default 50)

### Execution

To execute the desired configuration run python simo\simo.py

## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**
