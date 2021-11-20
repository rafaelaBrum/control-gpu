import argparse

from control.domain.instance_type import InstanceType
# from control.domain.task import Task

from control.util.loader import Loader

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

from control.config.logging_config import LoggingConfig

import logging

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


def test_on_demand_virtual_machine(number, loader):
    instance = InstanceType(
        provider=CloudManager.GCLOUD,
        instance_type='n2-standard-2',
        image_id='disk-ubuntu-flower-server',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031},
        vm_name=f'vm-teste-{number}'
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand',
        loader=loader
    )

    __prepare_logging()

    status = vm.deploy()

    if status:
        vm.prepare_vm()

        status = vm.terminate()

        if status:
            logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


# def main():
#     parser = argparse.ArgumentParser(description='Control GPU - v. 0.0.1')
#
#     parser.add_argument('--vm_number', required=True)
#     # parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
#     # parser.add_argument('--task_file', help="task file name", type=str, default=None)
#     # parser.add_argument('--env_file', help="env file name", type=str, default=None)
#     # parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
#     #
#     # parser.add_argument('--log_file', help="log file name", type=str, default=None)
#     #
#     # parser.add_argument('--revocation_rate',
#     #                     help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
#     #                     default=None)
#
#     parser.add_argument('--command', default='control')
#
#     loader = Loader(args=parser.parse_args())
#
#     print("Testing on demand VM")
#     test_on_demand_virtual_machine(args.vm_number, loader)
#
#
# if __name__ == "__main__":
#     main()
#     # print("Testing spot VM")
#     # test_preemptible_virtual_machine()
#     print("Test completed")
