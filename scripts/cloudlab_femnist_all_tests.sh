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
FEMNIST app (5 clients running 100 rounds)
---------------------------- "
echo " ----------------------------
Test Case 0 (without simulation)
---------------------------- "
cp setup_cloudlab/setup_femnist.cfg setup.cfg
COUNTER=4
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated
        python3 client.py control --emulated
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Test Case 1 (server and clients are faulty)
---------------------------- "
cp setup_cloudlab/setup_femnist_dyn_sched_all.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(1 hour)
 ---------------------------- "
COUNTER=4
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
COUNTER=3
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
cp setup_cloudlab/setup_femnist_dyn_sched_server.cfg setup.cfg
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
cp setup_cloudlab/setup_femnist_dyn_sched_cli.cfg setup.cfg
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

