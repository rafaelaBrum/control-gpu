#!/bin/bash
echo " ----------------------------
  Without FL
 ---------------------------- "
COUNTER=1
while [  $COUNTER -lt 4 ]; do
        echo " ----------------------------
  Running test - Counter=$COUNTER
 ----------------------------"
	FOLDER="results/$((COUNTER))_folder"
	time python3 main_app_without_fl.py -i -v --train -predst ~/IMGs-EN-194/trainset/ -split 0.9 0.10 0.00 -net Inception \
-data CellRep -d -e 50 -b 32 -tdim 240 240 -f1 10 -out logs/ -cpu 7 -gpu 0 -tn -wpath $FOLDER -model_dir $FOLDER \
-logdir $FOLDER -cache $FOLDER -test_dir ~/IMGs-EN-194/testset/ --pred
        COUNTER=$((COUNTER+1))
done
echo " ----------------------------
  Finished
 ---------------------------- "
