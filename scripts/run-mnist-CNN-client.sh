#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/mnist_client.py
PATH_DATASET=$HOME/control-gpu/data/MNIST/1/
python3 $CLIENT_FILE  -i -v --train -predst ~/.keras/datasets -split 0.857 0.013 0.13 -k -b 128 \
-out $HOME/control-gpu/logs/MN-flower -cpu 2 -gpu 0 -tn -wpath $HOME/control-gpu/results/MN-flower \
-model_dir $HOME/control-gpu/results/MN-flower -logdir $HOME/control-gpu/results/MN-flower \
-server_address $SERVER_ADDRESS -path_dataset $PATH_DATASET
