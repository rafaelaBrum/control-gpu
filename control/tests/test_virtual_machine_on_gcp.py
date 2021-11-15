from control.domain.instance_type import InstanceType
# from control.domain.task import Task

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


def test_on_demand_virtual_machine():
    instance = InstanceType(
        provider=CloudManager.GCLOUD,
        instance_type='n1-standard-1'
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand'
    )

    __prepare_logging()

    vm.deploy()

    vm.prepare_vm()

    status = vm.terminate()

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))

# if __name__ == "__main__":
#     print("Testing on demand VM")
#     test_on_demand_virtual_machine()
#     print("Testing spot VM")
#     test_preemptible_virtual_machine()
#     print("Test completed")
