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
SERVER_PROVIDER=gcp
SERVER_REGION=us-west1
SERVER_VM_NAME=e2-standard-4
CLIENTS_PROVIDER="aws aws aws aws"
CLIENTS_REGION="us-west-2 us-west-2 us-west-2 us-west-2"
CLIENTS_VM_NAME="g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge g4dn.2xlarge"
COUNTER=1
echo " ----------------------------
   Server in $SERVER_PROVIDER, $SERVER_REGION in a $SERVER_VM_NAME
  ----------------------------"
echo " ----------------------------
   Clients in $CLIENTS_PROVIDER
   $CLIENTS_REGION
   in $CLIENTS_VM_NAME
  ----------------------------"
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
