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
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/bustling-icon-331608-97742b8ca898.json
FOLDER_TESTS=input/FederatedLearning/mathematical_formulation/Inception_DS_AWS
COUNTER=1
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  cp $FOLDER_TESTS/presched_without_training_times.json $FOLDER_TESTS/presched.json
  echo python3 client.py pre --server_provider None --server_region None
  python3 client.py pre --server_provider None --server_region None
  cp $FOLDER_TESTS/presched.json $FOLDER_TESTS/presched_${COUNTER}.json
  cp $FOLDER_TESTS/presched_without_training_times.json $FOLDER_TESTS/presched.json
  echo " ----------------------------
   Sleeping
  ----------------------------"
  sleep 5m
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "

