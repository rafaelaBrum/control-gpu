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
for num_seed in 1 2 3 4 5 6 7 8 9 10
do
  echo " ----------------------------
  Seed $num_seed
  ---------------------------- "
  SERVER_PROVIDER="gcp"
  SERVER_REGION="us-central1"
  SERVER_VM_NAME="e2-standard-4"
  CLIENTS_PROVIDER="gcp gcp gcp gcp"
  CLIENTS_REGION="us-central1 us-central1 us-central1 us-central1"
  CLIENTS_VM_NAME="n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4 n1-standard-8_t4"
  strategy="FedAvg"
  echo " ----------------------------
     Server in $SERVER_PROVIDER, $SERVER_REGION in $SERVER_VM_NAME
    ----------------------------"
  echo " ----------------------------
     Clients in $CLIENTS_PROVIDER
     $CLIENTS_REGION
     in $CLIENTS_VM_NAME
    ----------------------------"
  echo python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME --strategy $strategy \
   --num_seed $num_seed
  python client.py control --server_provider $SERVER_PROVIDER --server_region $SERVER_REGION --server_vm_name $SERVER_VM_NAME \
   --clients_provider $CLIENTS_PROVIDER --clients_region $CLIENTS_REGION --clients_vm_name $CLIENTS_VM_NAME --strategy $strategy \
   --num_seed $num_seed
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "
