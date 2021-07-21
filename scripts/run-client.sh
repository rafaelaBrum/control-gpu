#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=/home/rafaelabrum/control-gpu/control/flower_tests/fcube_client.py
DATASET_PATH=/home/rafaelabrum/control-gpu/data/generated/3/
python3 $CLIENT_FILE --server_address $SERVER_ADDRESS --path_dataset $DATASET_PATH --batch-size 64