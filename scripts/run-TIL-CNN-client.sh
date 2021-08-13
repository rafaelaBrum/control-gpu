#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/TIL_Inception_client.py
PATH_DATASET=$HOME/IMGs-EN-194/1/

rm results/* -r

python3 $CLIENT_FILE  -i -v --train -predst $PATH_DATASET -split 0.9 0.1 0.0 -d -b 32 -tn \
-out $HOME/control-gpu/logs/TIL-flower/ -cpu 4 -gpu 0 -wpath $HOME/control-gpu/results/TIL-flower/ \
-model_dir $HOME/control-gpu/results/TIL-flower/ -logdir $HOME/control-gpu/results/TIL-flower/ \
-server_address $SERVER_ADDRESS -tdim 240 240 -f1 10 -cache $HOME/control-gpu/results/TIL-flower \
-test_dir $HOME/IMGs-EN-194/testset/
