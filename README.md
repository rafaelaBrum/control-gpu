# Multi-FedLS framework (version with Flower 1.30)

## Before you begin

### Dependencies

Multi-FedLS needs **Docker** to execute its database and at least **Python 3.10**.

### Repository structure

```
.
├── control/                        # Source code
├── extra_files/                    # Some utility Python source codes
├── gurobi_examples/                # Initial Gurobi tests
├── initial_setup/                  # DB and VMs setup files
├── input/                          # .JSON files from the environment and application
├── scripts/                        # Scripts to run experiments in batch mode
├── scripts_FL_app/                 # Initial local FL experiments
├── setup_files/                    # Configuration files needed to execute in batch mode
├── .gitignore
├── README.md                       # This file
├── client.py                       # Entry point of the framework
├── config.toml                     # File needed by the FL server VM to execute Flower
├── environment.env                 # Environmental variables needed by Multi-FedLS
├── gcloud_flower_commands.sh       # Sample commands to execute Flower and access VMs in GCP
├── requirements.txt                # Requirements file to execute Multi-FedLS (install in local machine)
├── requirements_client_flower.txt  # Requirements file to execute Daemon and Flower client (install via script in client AMI)
├── requirements_daemon.txt         # Requirements file to execute Daemon (duplicated)
├── requirements_server_flower.txt  # Requirements file to execute Daemon and Flower server (install via script in server AMI)
├── setup_dummy.cfg                 # Explaination of each field in setup.cfg file
└── setup.cfg                       # Configuration file read by Multi-FedLS in runtime
```

<!-- There is README files in each folder with further explaination. -->

### Cloud configuration Requirements

To execute Multi-FedLS you will need to configure two AMIs in each provider in each region with the scripts in `setup/vm_setup` folder. One AMI to the execute the clients and another to execute the server. If there is an option to create multi-region AMIs, it can be done. Add the respective AMIs id to each region in the `locations.json` file under the `input` folder.

#### Google Cloud Platform

To access GCP through Multi-FedLS, you will need to configure `gcloud` in your computer and create an Service Account with Editor permission and download a .json file with its key to your computer. The path to this json file needs to be updated in the `environment.env` file.

#### Amazon Web Service

*Under construction*

### Database and virtual environment setup

The first thing to do after downloading or cloning the repository in this branch (`new_paper`) is to execute the script `install_postgres.sh` localized in the `initial_setup` folder. Sometimes it is needed to run twice, as the container may not be up yet.

There is a command to ensure the DB container is up:
```
docker start pg-gpu-docker
```

After that, create a `virtualenv` via the following command:

```
python3.10 -m venv venv
```

This virtual environment needs to be activated every time Multi-FedLS will be executed. It is activated via the following command:

```
source venv/bin/activate
```

Now, the python dependencies need to be installed:

```
pip install -U pip
pip install -r requirements.txt
```

Then, we need to execute the `recreate_db` option in our framework:
```
source environment.env
python client.py recreate_db
```

## Execution setup

Multi-FedLS reads the `setup.cfg` file containing several configurations of the input, GCP, AWS, checkpointing, fault simulation and others. The file `dummy_setup.cfg` explains each field to be updated. 

## Execute Multi-FedLS

Before executing, make sure the DB container is up, the virtual environment is activated and the nedded environment variables are declared:
```
docker start pg-gpu-docker
source venv/bin/activate
source environment.env
```

There are 3 modes to execute Multi-FedLS. 

__(1)__ Display scheduling and app information:

```
python client.py info
```

__(2)__ Execute only the Pre-Scheduling module:
```
python client.py pre
```

__(3)__ Execute the whole framework:
```
python client.py control
```


In the last one, there are several possible command-line arguments to replace the values in the setup file. Please use `python client.py --help` to see all.