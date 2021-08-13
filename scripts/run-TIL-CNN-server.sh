#!/bin/bash
SERVER_FILE=$HOME/control-gpu/control/daemon/TIL_Inception_server.py
N_CLIENTS=2
python3 $SERVER_FILE --rounds 5 --sample_fraction 1 --min_sample_size $N_CLIENTS  --min_num_clients $N_CLIENTS
