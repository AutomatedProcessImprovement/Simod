Installation Guide
==================

This guide provides instructions on how to install SIMOD using **pip** (PyPI) or **Docker**.

Prerequisites
-------------
Before installing SIMOD, ensure you have the following dependencies:

Dependencies for local installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Python 3.9, 3.10, or 3.11**: The recommended version (extensively tested) is Python 3.9, however, it also works for
  Python versions 3.10 and 3.11.
- **Java 1.8**: Ensure Java is installed and added to your system’s PATH (e.g.,
  `Java.com <https://www.java.com/en/download/manual.jsp>`_).
- **Rust and Cargo (\*)**: If you are on a system without precompiled dependencies, you may also need to compile Rust
  and Cargo (install them using `rustup.rs <https://rustup.rs/>`_).

Dependencies for Docker installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Docker**: If you want to run SIMOD without installing dependencies, you can use the official Docker image (install
  Docker from `https://www.docker.com/get-started/ <https://www.docker.com/get-started/>`_).

Installation via PyPI
---------------------
The simplest way to install SIMOD is via **pip** from PyPI (`simod project <https://pypi.org/project/simod/>`_):

.. code-block:: bash

   python -m pip install simod

Running SIMOD after installation:

.. code-block:: bash

   simod --help

Installation via Docker
-----------------------
If you prefer running SIMOD inside a **Docker container**, in an isolated environment without requiring Python or Java
installations, use the following commands:

.. code-block:: bash

   docker pull nokal/simod

To start a container:

.. code-block:: bash

   docker run -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash

Use the `resources/` directory to store event logs and configuration files. The `outputs/` directory will contain the
results of SIMOD.

From inside the container, you can run SIMOD with:

.. code-block:: bash

   poetry run simod --help

Docker images for different SIMOD versions are available at `https://hub.docker.com/r/nokal/simod/tags <https://hub.docker.com/r/nokal/simod/tags>`_

Installation via source code
----------------------------
If you prefer to download the source code and compile it directly (you would need to have `git`, `python`, and
`poetry` installed), use the following commands:

.. code-block:: bash

   git clone https://github.com/AutomatedProcessImprovement/Simod.git

   cd Simod

   python -m venv simod-env

   # source ./simod-env/Scripts/activate  # for Linux systems
   .\simod-env\Scripts\activate.bat

   poetry install

Running SIMOD after installation:

.. code-block:: bash

   simod --help
