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
#for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="centralized_50_epochs_$((COUNTER))_folder_${instance_type}_4xlarge"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 50
    python3  test_CNN_TIL_without_fl_cloud --folder $FOLDER --n_parties 2 --instance_type $instance_type --rounds 50
    #sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  25 epochs
 ---------------------------- "
# for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
for instance_type in g4dn.xlarge
#for instance_type in g4dn.xlarge c5d.2xlarge r5dn.xlarge d3.xlarge
do
  COUNTER=1
  while [  $COUNTER -lt 4 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="centralized_50_epochs_$((COUNTER))_folder_${instance_type}_4xlarge"
    echo python3 test_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --rounds 25
    python3  test_CNN_TIL_without_fl_cloud --folder $FOLDER --n_parties 2 --instance_type $instance_type --rounds 25
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