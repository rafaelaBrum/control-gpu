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
#echo " ----------------------------
#4 clients running 30 rounds (server ckpt in aws)
#---------------------------- "
echo " ----------------------------
2 clients running 30 rounds (server ckpt in aws)
---------------------------- "
echo " ----------------------------
Test Case 0 (without simulation)
---------------------------- "
cp setup_jpdc_2023/test_case_3_setup_bucket_aws.cfg setup.cfg
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control
        python3 client.py control
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Test Case 1 (server and clients are faulty)
---------------------------- "
cp setup_jpdc_2023/test_case_3_setup_bucket_aws_all_revoking.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(2 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        python3 client.py control --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
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
        echo python3 client.py control --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        python3 client.py control --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
        fi
        COUNTER=$((COUNTER+1))
done
#echo " ----------------------------
#4 clients running 30 rounds (server ckpt in gcp)
#---------------------------- "
echo " ----------------------------
2 clients running 30 rounds (server ckpt in gcp)
---------------------------- "
echo " ----------------------------
Test Case 0 (without simulation)
---------------------------- "
cp setup_jpdc_2023/test_case_3_setup_bucket_gcp.cfg setup.cfg
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control
        python3 client.py control
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
        fi
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Test Case 1 (server and clients are faulty)
---------------------------- "
cp setup_jpdc_2023/test_case_3_setup_bucket_gcp_all_revoking.cfg setup.cfg
echo " ----------------------------
  Failure rate of 1/(2 hours)
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
        echo python3 client.py control --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        python3 client.py control --strategy FedAvgSave --revocation_rate 0.0001388888888888889
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
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
        echo python3 client.py control --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        python3 client.py control --strategy FedAvgSave --revocation_rate 6.944444444444444e-05
        if [ $COUNTER -ne 3 ]; then
          sleep 30m
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

