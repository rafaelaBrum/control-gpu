#!/bin/bash
SERVER_FILE=$HOME/control-gpu/control/daemon/mnist_server.py
python3 $SERVER_FILE --rounds 2 --sample_fraction 1 --min_sample_size 2 --min_num_clients 2
