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
echo " ----------------------------
4 clients running 50 rounds
---------------------------- "
COUNTER=1
echo " ----------------------------
Test Case 1 - clients checkpointing
---------------------------- "
cp setup_cloudlab/4_clients_50_rounds_ckpt_client.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --emulated --strategy FedAvg
  python3 client.py control --emulated --strategy FedAvg
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "
