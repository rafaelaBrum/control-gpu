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
for instance_type in g4dn.xlarge p2.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="2_clients_$((COUNTER))_folder_${instance_type}_heterogeneous"
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 2 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 2 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    # sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  3 clients
 ---------------------------- "
for instance_type in g4dn.xlarge p2.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
    FOLDER="3_clients_$((COUNTER))_folder_${instance_type}_heterogeneous"
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 3 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 3 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    # sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  4 clients
 ---------------------------- "
for instance_type in g4dn.xlarge p2.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="4_clients_$((COUNTER))_folder_${instance_type}_heterogeneous"
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds 5 --data_folder data_heterogeneous
    # if [ $COUNTER -ne 3 ]; then
    #   sleep 30m
    # fi
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "
