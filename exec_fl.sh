#!/bin/bash
echo " ----------------------------
  Starting docker
 ----------------------------"
docker start pg-gpu-docker
echo " ----------------------------
  Exporting variables
 ----------------------------"
export SETUP_FILE=setup.cfg
export SETUP_PATH=$HOME/control-gpu/
export NOTIFY_PWD='R1357908642@'
export POSTGRES_USER=postgres
export POSTGRES_PASS=rafaela123
echo " ----------------------------
 2 clients
 ---------------------------- "
for instance_type in g4dn.xlarge r5dn.xlarge
#for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="2_clients_$((COUNTER))_folder_${instance_type}"
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 2 --instance_type $instance_type
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 2 --instance_type $instance_type
    # sleep 5m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  3 clients
 ---------------------------- "
# for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
for instance_type in g4dn.xlarge r5dn.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER=3_clients_$((COUNTER))_folder_${instance_type}
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 3 --instance_type $instance_type
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 3 --instance_type $instance_type
    # sleep 5m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  4 clients
 ---------------------------- "
# for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
for instance_type in g4dn.xlarge r5dn.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER=4_clients_$((COUNTER))_folder_${instance_type}
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type
    # sleep 5m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  Stopping docker
 ----------------------------"
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ----------------------------"

