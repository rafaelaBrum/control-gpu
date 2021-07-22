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
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        python3 client.py control
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

