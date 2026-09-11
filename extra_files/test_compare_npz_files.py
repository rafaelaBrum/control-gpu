import numpy as np

import flwr as fl

from control.config.logging_config import LoggingConfig
from control.util.loader import Loader

import logging
import argparse

from pathlib import Path


def __prepare_logging():
    """
    Set up the log format, level and the file where it will be recorded.
    """
    logging_conf = LoggingConfig()
    log_file = Path(logging_conf.path, logging_conf.log_file)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_conf.level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description='Creating a t2.micro instance to check EBS content')
    parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
    parser.add_argument('--job_file', help="Job file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    parser.add_argument('--loc_file', help="loc file name", type=str, default=None)
    parser.add_argument('--pre_file', help="pre scheduling file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
    # parser.add_argument('--ac_size_seconds', help="Define the size of the Logical Allocation Cycle (seconds)",
    #                     type=int, default=None)

    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)
    parser.add_argument('--command', help='command para o client', type=str, default='')
    parser.add_argument('--num_clients_pre_scheduling', help="Quantity of clients in the pre-scheduling RPC tests",
                        type=int, default=None)

    parser.add_argument('--server_provider', help="Server provider", type=str, default=None, required=False)
    parser.add_argument('--server_region', help="Server region", type=str, default=None, required=False)
    parser.add_argument('--server_vm_name', help="Server VM name", type=str, default=None, required=False)

    parser.add_argument('--clients_provider', help="Each client provider", type=str, nargs='+', required=False)
    parser.add_argument('--clients_region', help="Each client region", type=str, nargs='+', required=False)
    parser.add_argument('--clients_vm_name', help="Each client VM name", type=str, nargs='+', required=False)

    parser.add_argument('--strategy', help="Each client VM name", type=str, default=None, required=False)

    parser.add_argument('--num_seed', help="Seed to be used by the clients to randomly shuffle their dataset",
                        default=None, required=False)
    parser.add_argument('--emulated', action='store_true', dest='emulated',
                        help='Using CloudLab to execute experiments', default=False)

    parser.add_argument('file_1', help="First file to compare", type=str)
    parser.add_argument('file_2', help="Second file to compare", type=str)

    args = parser.parse_args()
    loader = Loader(args=args)

    __prepare_logging()

    print(f"Loading {args.file_1} file")
    aux_data = np.load(args.file_1)
    aux_list: fl.common.NDArrays = []
    for file in aux_data.files:
        aux_list.append(aux_data[file])
    weights_1 = aux_list
    # print("weights_1")
    # print(weights_1)

    print(f"Loading {args.file_2} file")
    aux_data = np.load(args.file_2)
    aux_list: fl.common.NDArrays = []
    for file in aux_data.files:
        aux_list.append(aux_data[file])
    weights_2 = aux_list
    # print("weights_2")
    # print(weights_2)

    # for w_1 in weights_1:
    #     for w_2 in weights_2:
    #         comparison = w_1 == w_2
    #         equal_arrays = comparison.all()
    #         print(equal_arrays)

    # comparison = weights_1 == weights_2
    # equal_arrays = comparison.all()

    equal_arrays = all([np.allclose(x, y) for x, y in zip(weights_1, weights_2)])
    print(f"The weights are equal? {equal_arrays}")

    differences = [abs(x-y) for x, y in zip(weights_1, weights_2)]
    # print(differences)
    diff = 0.01
    minor_differences = [x <= diff for x in differences]
    # print(minor_differences)
    # number_of_false_values = [np.size(x) - np.count_nonzero(x) for x in minor_differences]
    # print(number_of_false_values)
    have_diff = all(x.all() for x in minor_differences)
    print(f"All differences is below {diff}: {have_diff}")


if __name__ == "__main__":
    main()
