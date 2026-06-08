#!/bin/bash
echo " ----------------------------
  Without FL
 ---------------------------- "
FOLDER="results/"
NET="VGG16"
TRAINSET="bucket_folder/data/CellRep/4_clients/0/trainset/"
TESTSET="bucket_folder/data/CellRep/4_clients/0/testset/"
EPOCHS=5
time python3 main_training.py -i -v -predst $TRAINSET -split 0.9 0.10 0.00 -net $NET -data CellRep -d -e $EPOCHS -b 32 -tdim 240 240 -out logs/ -cpu 4 -gpu 1 -tn -wpath $FOLDER -model_dir $FOLDER -logdir $FOLDER -cache $FOLDER -test_dir $TESTSET
echo " ----------------------------
  Finished
 ---------------------------- "
