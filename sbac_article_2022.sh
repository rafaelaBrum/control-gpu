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
Scenario 1 (all DS in AWS east)
---------------------------- "
cp setup_sbac_2022/test_case_1_setup.cfg setup.cfg
echo " ----------------------------
Optimal solution
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-west-2"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-west-2 us-west-2 us-west-2 us-west-2"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #1
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-east-1"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-east-1 us-east-1 us-east-1 us-east-1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #2
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="gcp gcp gcp gcp"
CLIENTS_REGION="us-central1 us-central1 us-central1 us-central1"
CLIENTS_VM_NAME="n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Scenario 2 (all DS in GCP central)
---------------------------- "
cp setup_sbac_2022/test_case_2_setup.cfg setup.cfg
echo " ----------------------------
Optimal solution
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="gcp gcp gcp gcp"
CLIENTS_REGION="us-central1 us-central1 us-central1 us-central1"
CLIENTS_VM_NAME="n1-standard-8_v100 n1-standard-8_v100 n1-standard-8_v100 n1-standard-8_v100"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #1
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="gcp gcp gcp gcp"
CLIENTS_REGION="us-central1 us-central1 us-central1 us-central1"
CLIENTS_VM_NAME="n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #2
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-east-1"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-east-1 us-east-1 us-east-1 us-east-1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Scenario 3 (2 DS in AWS east and 2 DS in GCP central)
---------------------------- "
cp setup_sbac_2022/test_case_3_setup.cfg setup.cfg
echo " ----------------------------
Optimal solution
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-west-2"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-west-2 us-west-2 us-west-2 us-east-1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #1
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-east-1"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws gcp gcp"
CLIENTS_REGION="us-east-1 us-east-1 us-central1 us-central1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #2
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="aws aws gcp gcp"
CLIENTS_REGION="us-east-1 us-east-1 us-central1 us-central1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Scenario 4 (3 DS in AWS east and 1 DS in GCP central)
---------------------------- "
cp setup_sbac_2022/test_case_4_setup.cfg setup.cfg
echo " ----------------------------
Optimal solution
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-west-2"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-west-2 us-west-2 us-west-2 us-east-1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #1
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-east-1"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws gcp"
CLIENTS_REGION="us-east-1 us-east-1 us-east-1 us-central1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #2
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-east-1"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-east-1 us-east-1 us-east-1 us-east-1"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Scenario 5 (1 DS in AWS east and 3 DS in GCP central)
---------------------------- "
cp setup_sbac_2022/test_case_5_setup.cfg setup.cfg
echo " ----------------------------
Optimal solution
---------------------------- "
SERVER_PROVIDER="aws"
SERVER_REGION="us-west-2"
SERVER_VM_NAME="t2.xlarge"
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-west-2 us-west-2 us-east-1 us-west-2"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=1
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #1
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="aws gcp gcp gcp"
CLIENTS_REGION="us-east-1 us-central1 us-central1 us-central1"
CLIENTS_VM_NAME="g4dn.2xlarge n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
    sleep 30m
  fi
  COUNTER=$((COUNTER+1))
done
echo " ----------------------------
Random test #2
---------------------------- "
SERVER_PROVIDER="gcp"
SERVER_REGION="us-central1"
SERVER_VM_NAME="e2-standard-4"
CLIENTS_PROVIDER="gcp gcp gcp gcp"
CLIENTS_REGION="us-central1 us-central1 us-central1 us-central1"
CLIENTS_VM_NAME="n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4"
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
COUNTER=4
while [  $COUNTER -lt 4 ]; do
  echo " ----------------------------
   Running test - Counter=$COUNTER
  ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME
  if [ $COUNTER -ne 3 ]; then
    echo " ----------------------------
   Sleeping
  ----------------------------"
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
