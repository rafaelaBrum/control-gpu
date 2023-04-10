#!/bin/bash
SERVER_FILE=$HOME/control-gpu/control/daemon/TF_femnist_server.py
N_CLIENTS=1
STRATEGY='FedAvg'

python3 $SERVER_FILE --rounds 400 --sample_fraction 1 --min_sample_size $N_CLIENTS \
 --min_num_clients $N_CLIENTS --strategy $STRATEGY
