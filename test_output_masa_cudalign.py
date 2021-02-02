from control.domain.instance_type import InstanceType
# from control.domain.task import Task

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

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
    parser.add_argument('--task_file', help="task file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    # parser.add_argument('--map_file', help="map file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
    # parser.add_argument('--ac_size_seconds', help="Define the size of the Logical Allocation Cycle (seconds)",
    #                     type=int, default=None)

    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)
    parser.add_argument('--command', help='command para o client', type=str, default='')
    parser.add_argument('volume_id', help="Volume id to be attached", type=str)

    args = parser.parse_args()
    loader = Loader(args=args)
    volume_id = args.volume_id

    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-07ae9c26b070d6a66',
        ebs_device_name='/dev/xvdf',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031}
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='preemptible',
        loader=loader
    )

    __prepare_logging()

    if volume_id is not None:
        vm.volume_id = volume_id

    vm.deploy()

    vm.prepare_vm()


if __name__ == "__main__":
    main()
