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
  4 clients
 ---------------------------- "
for instance_type in g4dn.xlarge
do
  for rounds in 5 10
  do
    COUNTER=1
    while [  $COUNTER -lt 2 ]; do
      echo " ----------------------------
       Running test - Counter=$COUNTER
      ----------------------------"
      FOLDER="4_clients_${rounds}_rounds_30_epochs_$((COUNTER))_folder_${instance_type}"
      echo python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds $rounds --epochs 30 --data_folder data
      python3 test_CNN_TIL_cloud_environment.py --folder $FOLDER --n_parties 4 --instance_type $instance_type --rounds $rounds --epochs 30 --data_folder data
      # if [ $COUNTER -ne 3 ]; then
      #   sleep 30m
      # fi
      COUNTER=$((COUNTER+1))
    done
  done
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "
