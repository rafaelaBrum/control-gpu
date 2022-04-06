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
FOLDER_TESTS=input/FederatedLearning/rpc_concurrent_tests
COUNTERCONC=4
while [  $COUNTERCONC -lt 5 ]; do
    echo " ----------------------------
  Running test - Num. clients=$COUNTERCONC
 ----------------------------"
    COUNTER=1
    while [  $COUNTER -lt 11 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        cp $FOLDER_TESTS/presched_without_values.json $FOLDER_TESTS/presched.json
        python client.py pre --num_clients $COUNTERCONC
        cp $FOLDER_TESTS/presched.json $FOLDER_TESTS/presched_${COUNTERCONC}_clients_${COUNTER}.json
        cp $FOLDER_TESTS/presched_without_values.json $FOLDER_TESTS/presched.json
        echo " ----------------------------
  Sleeping
 ----------------------------"
        if [  $COUNTER -lt 10 ]; then
                sleep 30m
        fi
        COUNTER=$((COUNTER+1))
    done
    COUNTERCONC=$((COUNTERCONC+1))
done
echo " ----------------------------
  Stopping docker
 ----------------------------"
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ----------------------------"

