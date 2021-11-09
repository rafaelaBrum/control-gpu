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
  4 clients - 50 epochs in total
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  COUNTER=1
  epochs=2
  rounds=25
  echo " ----------------------------
  $rounds comm rounds - $epochs epochs per round
 ---------------------------- "
  while [  $COUNTER -lt 3 ]; do
    echo " ----------------------------
     Running test - Counter=$COUNTER
    ----------------------------"
    FOLDER="50_epochs_total_4_clients_${epochs}_epochs_$((COUNTER))_folder_${instance_type}"
    echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds $rounds --epochs $epochs --data_folder data
    python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds $rounds --epochs $epochs --data_folder data
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
