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
FOLDER_TESTS=input/FederatedLearning/rpc_tests
COUNTER=1
while [  $COUNTER -lt 6 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        cp $FOLDER_TESTS/presched_sem_rpc.json $FOLDER_TESTS/presched.json
        python client.py pre
        cp $FOLDER_TESTS/presched.json $FOLDER_TESTS/presched_$COUNTER.json
        cp $FOLDER_TESTS/presched_sem_rpc.json $FOLDER_TESTS/presched.json
        echo " ----------------------------
  Sleeping
 ----------------------------"
        if [  $COUNTER -lt 5 ]; then
                sleep 4h
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Stopping docker
 ----------------------------"
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ----------------------------"

