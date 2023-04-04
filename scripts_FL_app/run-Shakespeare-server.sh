#!/bin/bash
SERVER_FILE=$HOME/control-gpu/control/daemon/TF_shakespeare_server.py
N_CLIENTS=1
python3 $SERVER_FILE --rounds 20 --sample_fraction 1 --min_sample_size $N_CLIENTS  --min_num_clients $N_CLIENTS
