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
 20 epochs
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=2
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="centralized_20_epochs_$((COUNTER))_folder_${instance_type}_ondemand"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 20
    python3  test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 20
    #sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  30 epochs
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=2
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="centralized_30_epochs_$((COUNTER))_folder_${instance_type}_ondemand"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 30
    python3  test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 30
    #sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  40 epochs
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=2
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
     Running test - Counter=$COUNTER
    ----------------------------"
    FOLDER="centralized_40_epochs_$((COUNTER))_folder_${instance_type}_ondemand"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 40
    python3  test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 40
    #sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  50 epochs
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=2
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
     Running test - Counter=$COUNTER
    ----------------------------"
    FOLDER="centralized_50_epochs_$((COUNTER))_folder_${instance_type}_ondemand"
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
