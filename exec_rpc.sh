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
 RPC tests
 ---------------------------- "
FOLDER_TESTS=input/FederatedLearning/teste_rpc
COUNTER=1
while [  $COUNTER -lt 6 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        python client.py pre
        cp $FOLDER_TESTS/presched.json $FOLDER_TESTS/presched_$COUNTER.json
        sleep 5m
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Stopping docker
 ----------------------------"
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ----------------------------"

