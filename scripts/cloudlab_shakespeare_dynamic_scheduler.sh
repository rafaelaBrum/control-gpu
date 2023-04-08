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
Shakespeare app (8 clients running 20 rounds)
---------------------------- "
echo " ----------------------------
Test Case 1 (server and clients are faulty)
---------------------------- "
cp setup_cloudlab/setup_shakespeare_dyn_sched_all.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(1 hour)
 ---------------------------- "
COUNTER=2
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Failure rate of 1/(2 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Test Case 2 (only server is faulty)
---------------------------- "
cp setup_cloudlab/setup_shakespeare_dyn_sched_server.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(1 hour)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Failure rate of 1/(2 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Test Case 3 (only clients are faulty)
---------------------------- "
cp setup_cloudlab/setup_shakespeare_dyn_sched_server.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(1 hour)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.00027777777
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Failure rate of 1/(2 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
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

