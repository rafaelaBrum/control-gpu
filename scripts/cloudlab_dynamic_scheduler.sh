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
cp setup_cloudlab/4_clients_50_rounds_dynamic_scheduler.cfg setup.cfg
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
  Failure rate of 1/(4 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
	echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        if [ $COUNTER -ne 3 ]; then
          sleep 5m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Failure rate of 1/(6 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
	echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 4.6296296296296294e-05
        python3 client.py control --emulated --strategy FedAvgSave --revocation_rate 4.6296296296296294e-05
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

