import grpc
import logging

import numpy as np
import json
import argparse

from dummy_grpc import dummy_pb2_grpc
from dummy_grpc.dummy_parameters import weights_to_parameters
from dummy_grpc.dummy_serde import fit_ins_to_proto, evaluate_ins_to_proto, GRPC_MAX_MESSAGE_LENGTH, scalar_to_proto
from dummy_grpc.dummy_typing import FitIns, EvaluateIns

from time import time


def finish(stub):
    request = scalar_to_proto('t')
    logging.info("Sending Finish message")
    stub.Finish(request=request)
    logging.info("Finishing Finish message")


def testing_eval(stub, length_parameters):
    logging.info("Initializing TestEval message")
    weights = [np.zeros(length_parameters)]
    parameters = weights_to_parameters(weights=weights)
    request = evaluate_ins_to_proto(EvaluateIns(parameters=parameters, config={'test': 'test'}))
    logging.info("Sending TestEval message")
    stub.TestEval(request=request)
    logging.info("Finishing TestEval message")


def testing_fit(stub, length_parameters):
    logging.info("Initializing TestTrain message")
    weights = [np.zeros(length_parameters)]
    parameters = weights_to_parameters(weights=weights)
    request = fit_ins_to_proto(FitIns(parameters=parameters, config={'test': 'test'}))
    logging.info("Sending TestTrain message")
    stub.TestTrain(request=request)
    logging.info("Finishing TestTrain message")


def run(server_address, file, length_parameters):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    times = {}
    logging.info("Initializing dummy client")
    with grpc.insecure_channel(server_address,
                               options=[
                                   ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                                   ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH)
                               ],) as channel:
        stub = dummy_pb2_grpc.DummyServiceStub(channel)
        print("-------------- TestTrain --------------")
        t1 = time()
        testing_fit(stub=stub, length_parameters=length_parameters)
        t2 = time()
        times['TrainMsg'] = (str(t2 - t1))
        logging.info(times['TrainMsg'])
        print("-------------- TestEval --------------")
        t1 = time()
        testing_eval(stub=stub, length_parameters=length_parameters)
        t2 = time()
        times['TestMsg'] = str(t2 - t1)
        logging.info(times['TestMsg'])
        print("-------------- Finish --------------")
        finish(stub=stub)

        print("times")
        print(times)

        with open(file, 'w') as f:
            f.write(json.dumps(times))


if __name__ == '__main__':
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel('INFO')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    parser = argparse.ArgumentParser(description='Running Flower client and getting the message exchange time')

    parser.add_argument('--server_address', help="Flower server address", type=str, default=None, required=True)
    parser.add_argument('--length_parameters', help="Size of parameter", type=int, default=None, required=True)
    parser.add_argument('--save_file', help="File to save results", type=str, default=None, required=True)

    args = parser.parse_args()
    address = args.server_address
    length = args.length_parameters
    save_file = args.save_file
    run(server_address=address, length_parameters=length, file=save_file)
