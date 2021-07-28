#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=/home/rafaelabrum/control-gpu/control/flower_tests/mnist_TF_client.py
python3 $CLIENT_FILE --server_address $SERVER_ADDRESS --batch-size 64