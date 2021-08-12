#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/TIL_Inception_client.py
PATH_DATASET=$HOME/IMGs-EN-194/1/

rm cache/*

python3 $CLIENT_FILE  -i -v --train -predst $PATH_DATASET -split 0.9 0.1 0.0 -k -b 32 -tn \
-out $HOME/control-gpu/logs/MN-flower -cpu 4 -gpu 0 -wpath $HOME/control-gpu/results/TIL-flower \
-model_dir $HOME/control-gpu/results/MN-flower -logdir $HOME/control-gpu/results/TIL-flower \
-server_address $SERVER_ADDRESS -tdim 240 240 -f1 10