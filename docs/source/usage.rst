Usage Guide
===========

This guide provides instructions on how to use SIMOD from command line to discover a BPS model out of an event log in
CSV format.

Running Simod
-------------

Once Simod is installed (see `Installation <installation.html>`_), you can run it by specifying a configuration file.

Installed via PyPI or source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   simod --configuration resources/config/configuration_example.yml

Replace `resources/config/configuration_example.yml` with the path to your own configuration file. Paths can be
relative to the configuration file or absolute.


Installed via Docker
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   poetry run simod --configuration resources/config/configuration_example.yml

Replace `resources/config/configuration_example.yml` with the path to your own configuration file. Paths can be
relative to the configuration file or absolute.

Configuration File
------------------
The configuration file is a YAML file that specifies various parameters for Simod. Ensure that the path to your event
log is specified in the configuration file. Here are some configuration examples:

- Basic configuration to discover the full BPS
  model (`basic <_static/configuration_example.yml>`_).
- Basic configuration to discover the full BPS model using fuzzy (probabilistic) resource
  calendars (`probabilistic <_static/configuration_example_fuzzy.yml>`_).
- Basic configuration to discover the full BPS model with data-aware branching rules
  (`data-aware <_static/configuration_example_data_aware.yml>`_).
- Basic configuration to discover the full BPS model, and evaluate it with a specified event
  log (`with evaluation <_static/configuration_example_with_evaluation.yml>`_).
- Basic configuration to discover a BPS model with a provided BPMN process model as starting
  point (`with BPMN model <_static/configuration_example_with_provided_process_model.yml>`_).
- Basic configuration to discover a BPS model with no optimization process (one-shot)
  (`one-shot <_static/configuration_one_shot.yml>`_).
- Complete configuration example with all the possible
  parameters (`complete config <_static/complete_configuration.yml>`_).

Event Log Format
----------------
Simod takes as input an event log in CSV format.

.. _tab_event_log:
.. table:: Sample of input event log format.
    :align: center

    =======  ===========  ===================  ===================  ========
    case_id  activity     start_time           end_time             resource
    =======  ===========  ===================  ===================  ========
    512      Create PO    03/11/2021 08:00:00  03/11/2021 08:31:11  DIO
    513      Create PO    03/11/2021 08:34:21  03/11/2021 09:02:09  DIO
    514      Create PO    03/11/2021 09:11:11  03/11/2021 09:49:51  DIO
    512      Approve PO   03/11/2021 12:13:06  03/11/2021 12:44:21  Joseph
    513      Reject PO    03/11/2021 12:30:51  03/11/2021 13:15:50  Jolyne
    514      Approve PO   03/11/2021 12:59:11  03/11/2021 13:32:36  Joseph
    512      Check Stock  03/11/2021 14:22:10  03/11/2021 14:49:22  DIO
    514      Check Stock  03/11/2021 15:11:01  03/11/2021 15:46:12  DIO
    514      Order Goods  04/11/2021 09:46:12  04/11/2021 10:34:23  Joseph
    512      Pack Goods   04/11/2021 10:46:50  04/11/2021 11:18:02  Giorno
    =======  ===========  ===================  ===================  ========

The column names can be specified as part of the configuration file (`see here <_static/complete_configuration.yml>`_).

Output
------
Simod discovers a business process simulation model that can be simulated using the
`Prosimos simulator <https://github.com/AutomatedProcessImprovement/Prosimos>`_, which is embedded in Simod.

Once SIMOD is finished, the discovered BPS model can be found in the `outputs` directory, under the folder `best_result`.
