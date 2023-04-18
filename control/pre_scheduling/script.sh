#!/bin/bash

ln -s /mydata/ data
python3.7 main_training_femnist.py -id_client 0
mv times.json times_0.json
python3.7 main_training_femnist.py -id_client 1
mv times.json times_1.json
python3.7 main_training_femnist.py -id_client 2
mv times.json times_2.json
python3.7 main_training_femnist.py -id_client 3
mv times.json times_3.json
python3.7 main_training_femnist.py -id_client 4
mv times.json times_4.json
