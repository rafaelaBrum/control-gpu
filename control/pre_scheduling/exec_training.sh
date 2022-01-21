#!/bin/bash
echo " ----------------------------
  Without FL
 ---------------------------- "
FOLDER="results/"
time python3 main_training.py -i -v --train -predst ~/bucket_folder/trainset/ -split 0.9 0.10 0.00 -net Inception \
-data CellRep -d -e 10 -b 32 -tdim 240 240 -f1 10 -met 30 -out logs/ -cpu 7 -gpu 1 -tn -wpath $FOLDER -model_dir $FOLDER \
-logdir $FOLDER -cache $FOLDER -test_dir ~/bucket_folder/testset/
echo " ----------------------------
  Finished
 ---------------------------- "
