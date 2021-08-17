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
  Without FL
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
	time python3 main_app_without_fl.py -i -v --pred -predst ~/IMGs-EN-194/trainset/ -split 0.9 0.10 0.00 -net Inception -data CellRep -d -e 100 -b 32 -tdim 240 240 -f1 10 -out logs/ -cpu 7 -gpu 0 -tn -wpath results/new_nds300/ -model_dir results/new_nds300/ -logdir results/new_nds300/ -cache results/new_nds300 -test_dir ~/IMGs-EN-194/testset/
        COUNTER=$((COUNTER+1))
done
echo " ---------------------------
  Stoping docker
 ------------------------------"
docker stop pg-gpu-docker
echo " ----------------------------
  Finished
 ----------------------------"

