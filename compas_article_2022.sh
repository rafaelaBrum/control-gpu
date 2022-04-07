#!/bin/bash
echo " ----------------------------
  Starting docker
 ----------------------------"
docker start pg-gpu-docker
echo " ----------------------------
  Exporting variables
 ----------------------------"
export SETUP_FILE=setup.cfg
export SETUP_PATH=$HOME/control-gpu_GCP/
export NOTIFY_PWD='R1357908642@'
export POSTGRES_USER=postgres
export POSTGRES_PASS=rafaela123
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/bustling-icon-331608-97742b8ca898.json
COUNTER=1
echo " ----------------------------
Test Case 1 (4 clients in AWS west - server in AWS west)
---------------------------- "
cp setup_compas_article/test_case_1_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider aws --server_region us-west-2
  python3 client.py control --server_provider aws --server_region us-west-2
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 2 (2 clients in AWS west, 2 clients in AWS east - server in AWS west)
---------------------------- "
cp setup_compas_article/test_case_2_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider aws --server_region us-west-2
  python3 client.py control --server_provider aws --server_region us-west-2
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 3 (2 clients in AWS west, 2 clients in AWS east - server in AWS east)
---------------------------- "
cp setup_compas_article/test_case_3_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider aws --server_region us-east-1
  python3 client.py control --server_provider aws --server_region us-east-1
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 4 (4 clients in GCP west - server in GCP west)
---------------------------- "
cp setup_compas_article/test_case_4_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider gcp --server_region us-west1
  python3 client.py control --server_provider gcp --server_region us-west1
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 5 (2 clients in GCP west, 2 clients in GCP east - server in GCP west)
---------------------------- "
cp setup_compas_article/test_case_5_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider gcp --server_region us-west1
  python3 client.py control --server_provider gcp --server_region us-west1
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 6 (2 clients in GCP west, 2 clients in GCP east - server in GCP east)
---------------------------- "
cp setup_compas_article/test_case_6_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider gcp --server_region us-east4
  python3 client.py control --server_provider gcp --server_region us-east4
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 7 (3 clients in AWS west, 1 clients in GCP west - server in AWS west)
---------------------------- "
cp setup_compas_article/test_case_7_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider aws --server_region us-west-2
  python3 client.py control --server_provider aws --server_region us-west-2
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 8 (2 clients in AWS west, 2 clients in GCP west - server in AWS west)
---------------------------- "
cp setup_compas_article/test_case_8_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider aws --server_region us-west-2
  python3 client.py control --server_provider aws --server_region us-west-2
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 9 (2 clients in AWS west, 2 clients in GCP west - server in GCP west)
---------------------------- "
cp setup_compas_article/test_case_9_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider gcp --server_region us-west1
  python3 client.py control --server_provider gcp --server_region us-west1
  if [ $COUNTER -ne 3 ]; then
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
COUNTER=1
echo " ----------------------------
Test Case 10 (1 client in AWS west, 3 clients in GCP west - server in GCP west)
---------------------------- "
cp setup_compas_article/test_case_10_setup.cfg setup.cfg
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python3 client.py control --server_provider gcp --server_region us-west1
  python3 client.py control --server_provider gcp --server_region us-west1
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
