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
 150 epochs
 ---------------------------- "
for instance_type in p2.xlarge
do
  COUNTER=2
  while [  $COUNTER -lt 3 ]; do
    echo " ----------------------------
     Running test - Counter=$COUNTER
    ----------------------------"
    FOLDER="centralized_once_150_epochs_${instance_type}_on_demand"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --epochs 150
    python3  test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --epochs 150
    #sleep 30m
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
