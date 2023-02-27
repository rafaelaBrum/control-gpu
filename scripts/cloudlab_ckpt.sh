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
COUNTER=4
echo " ----------------------------
Test Case 1 (every 10 rounds)
---------------------------- "
cp setup_cloudlab/4_clients_50_rounds_ckpt_10_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --emulated --strategy FedAvgSave
  python3 client.py control --emulated --strategy FedAvgSave
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=4
echo " ----------------------------
Test Case 2 (every 20 rounds)
---------------------------- "
cp setup_cloudlab/4_clients_50_rounds_ckpt_20_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --emulated --strategy FedAvgSave
  python3 client.py control --emulated --strategy FedAvgSave
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=3
echo " ----------------------------
Test Case 3 (every 30 rounds)
---------------------------- "
cp setup_cloudlab/4_clients_50_rounds_ckpt_30_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --emulated --strategy FedAvgSave
  python3 client.py control --emulated --strategy FedAvgSave
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 4 (every 40 rounds)
---------------------------- "
cp setup_cloudlab/4_clients_50_rounds_ckpt_40_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --emulated --strategy FedAvgSave
  python3 client.py control --emulated --strategy FedAvgSave
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
