#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/TIL_Inception_client.py
PATH_TRAIN_DATASET=$HOME/IMGs-EN-194/1/
PATH_TEST_DATASET=$HOME/IMGs-EN-194/testset/
RESULTS_PATH=$HOME/control-gpu/results/TIL-flower/
LOG_PATH=$HOME/control-gpu/logs/TIL-flower/

rm results/* -r

python3 $CLIENT_FILE  -i -v --train -predst $PATH_TRAIN_DATASET -split 0.9 0.1 0.0 -d -b 32 -tn \
-out $LOG_PATH -cpu 4 -gpu 0 -wpath $RESULTS_PATH \
-model_dir $RESULTS_PATH -logdir $RESULTS_PATH \
-server_address $SERVER_ADDRESS -tdim 240 240 -f1 10 -cache $RESULTS_PATH \
-test_dir $PATH_TEST_DATASET
