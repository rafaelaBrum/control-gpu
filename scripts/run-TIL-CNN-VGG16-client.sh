#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/TIL_VGG_client.py
PATH_TRAIN_DATASET=$HOME/IMGs-EN-194/trainset/
PATH_TEST_DATASET=$HOME/IMGs-EN-194/testset/
RESULTS_PATH=$HOME/control-gpu/results/TIL-flower-VGG/
LOG_PATH=$HOME/control-gpu/logs/TIL-flower-VGG/
EPOCHS=10

rm results/* -r

time python3 $CLIENT_FILE  -i -v --train -predst $PATH_TRAIN_DATASET -split 0.9 0.1 0.0 -d -b 64 -tn \
-out $LOG_PATH -cpu 5 -gpu 0 -wpath $RESULTS_PATH \
-model_dir $RESULTS_PATH -logdir $RESULTS_PATH \
-server_address $SERVER_ADDRESS -tdim 240 240 -f1 10 -cache $RESULTS_PATH \
-test_dir $PATH_TEST_DATASET -epochs $EPOCHS
