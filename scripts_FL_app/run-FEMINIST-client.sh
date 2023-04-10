#!/bin/bash
SERVER_ADDRESS=localhost:8080
CLIENT_FILE=$HOME/control-gpu/control/daemon/TF_femnist_client.py
ID_CLIENT=0

time python3 $CLIENT_FILE -id_client $ID_CLIENT -server_address $SERVER_ADDRESS