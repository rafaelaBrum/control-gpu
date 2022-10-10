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


def test_on_demand_virtual_machine(loader):
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-0c20212ac4ce26a60',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031},
        ebs_device_name='/dev/xvdf',
        count_gpu=0,
        gpu='',
        memory=0,
        vcpu=2,
        locations=''
    )

    # task = Task(
    #     task_id=2,
    #     memory=0.2,
    #     command="ls",
    #     io=0,
    #     runtime={'t2.nano': 100}
    # )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand',
        loader=loader
    )

    __prepare_logging()

    vm.deploy(type_task='server')

    vm.prepare_vm(type_task='server')

    input("Enter to continue with VM termination")

    status = vm.terminate()

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


def test_preemptible_virtual_machine(loader):
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-05967bab0d693d334',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031},
        count_gpu=0,
        gpu='',
        memory=0,
        vcpu=2,
        locations=''
    )

    # task = Task(
    #     task_id=2,
    #     memory=0.2,
    #     command="ls",
    #     io=0,
    #     runtime={'t2.nano': 100}
    # )

    vm = VirtualMachine(
        instance_type=instance,
        market='preemptible',
        loader=loader
    )

    __prepare_logging()

    vm.deploy(type_task='server')

    vm.prepare_vm(type_task='server')

    status = vm.terminate()

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


def test_vm_with_ebs(volume_id='', loader=None):
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-05967bab0d693d334',
        ebs_device_name='/dev/xvdf',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031},
        count_gpu=0,
        gpu='',
        memory=0,
        vcpu=2,
        locations=''
    )

    # task = Task(
    #     task_id=2,
    #     memory=0.2,
    #     command="ls",
    #     io=0,
    #     runtime={'t2.nano': 100}
    # )

    vm = VirtualMachine(
        instance_type=instance,
        market='preemptible',
        loader=loader
    )

    __prepare_logging()

    if volume_id is not None:
        vm.volume_id = volume_id

    vm.deploy(type_task='server')

    vm.prepare_vm(type_task='server')

    status = vm.terminate(delete_volume=False)

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))

# if __name__ == "__main__":
#     print("Testing on demand VM")
#     test_on_demand_virtual_machine()
#     print("Testing spot VM")
#     test_preemptible_virtual_machine()
#     print("Test completed")
