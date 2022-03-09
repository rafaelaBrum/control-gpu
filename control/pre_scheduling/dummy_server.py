from dummy_grpc.dummy_serde import fit_ins_from_proto, fit_res_to_proto, evaluate_ins_from_proto, \
    evaluate_res_to_proto, GRPC_MAX_MESSAGE_LENGTH
from dummy_grpc.dummy_typing import FitRes, EvaluateRes

from concurrent import futures
import logging

import grpc
from dummy_grpc import dummy_pb2_grpc

from time import sleep
import argparse


class DummyServicer(dummy_pb2_grpc.DummyServiceServicer):
    finish = False

    def TestTrain(self, request, context):
        logging.info("Received TestTrain message")
        message = fit_ins_from_proto(request)
        metrics = {"test": 10.0,
                   "test1": 10.0}
        return fit_res_to_proto(FitRes(parameters=message.parameters, num_examples=10, metrics=metrics))

    def TestEval(self, request, context):
        logging.info("Received TestEval message")
        evaluate_ins_from_proto(request)
        metrics = {"test": 10.0,
                   "test1": 10.0,
                   "test2": 10.0,
                   "test3": 10.0}
        return evaluate_res_to_proto(EvaluateRes(loss=0.1, num_examples=10, metrics=metrics))

    def Finish(self, request, context):
        logging.info("Received Finish message")
        self.finish = True
        return request

    def has_finished(self):
        return self.finish


def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[
                             ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                             ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
                         ],
                         )
    dummy_servicer = DummyServicer()
    dummy_pb2_grpc.add_DummyServiceServicer_to_server(
        dummy_servicer, server)
    server.add_insecure_port(address)
    logging.info("Initializing dummy server")
    server.start()
    while not dummy_servicer.has_finished():
        sleep(1)
    server.stop(grace=1)


if __name__ == '__main__':
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel('INFO')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='Running dummy Flower server')

    parser.add_argument('--fl_port', help="Flower server address", type=int, default=None, required=True)

    args = parser.parse_args()

    server_address = '[::]:' + str(args.fl_port)

    serve(address=server_address)
