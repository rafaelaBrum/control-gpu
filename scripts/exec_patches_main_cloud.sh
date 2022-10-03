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
 50 epochs
 ---------------------------- "
for instance_type in g4dn.2xlarge
do
  COUNTER=1
  neural_network="Inception"
  split="0.98 0.01 0.01"
  underscore_split=$(echo $split | sed 's/ /_/g')
  STR1=`echo $STR1 | sed 's/ /,/g'`
  dataset_folder='brca'
  while [  $COUNTER -lt 2 ]; do
    echo " ----------------------------
    Running test - Counter=$COUNTER
   ----------------------------"
   FOLDER="patches_${dataset_folder}_50_epochs_${neural_network}_split_${underscore_split}_$((COUNTER))_folder_${instance_type}"
    echo python3 test_patches_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --dataset_folder $dataset_folder --neural_network $neural_network --split $split --epochs 50
    python3 test_patches_CNN_TIL_without_fl_cloud.py --folder $FOLDER --instance_type $instance_type --dataset_folder $dataset_folder --neural_network $neural_network --split $split --epochs 50
    #sleep 30m
    COUNTER=$((COUNTER+1))
  done
done
echo " ----------------------------
  Stopping docker
 ---------------------------- "
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ---------------------------- "
