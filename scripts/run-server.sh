#!/bin/bash
SERVER_FILE=/home/rafaelabrum/control-gpu/control/flower_tests/fcube_server.py
python3 $SERVER_FILE --rounds 10 --sample_fraction 1 --min_sample_size 1 --min_num_clients 1