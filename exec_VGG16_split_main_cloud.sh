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
  50 epochs
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 3 ]; do
    echo " ----------------------------
     Running test - Counter=$COUNTER
    ----------------------------"
    FOLDER="centralized_split_09_01_$((COUNTER))_folder_${instance_type}_ondemand"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 50
    python3  test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 50
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
