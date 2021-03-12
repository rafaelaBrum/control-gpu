import logging

from control.config.communication_config import CommunicationConfig
from control.config.logging_config import LoggingConfig
from control.domain.instance_type import InstanceType

from control.domain.task import Task
from control.managers.cloud_manager import CloudManager
from control.managers.dispatcher import Executor
from control.managers.virtual_machine import VirtualMachine

from control.daemon.daemon_manager import Daemon
from control.daemon.communicator import Communicator

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


def __prepare_daemon(vm: VirtualMachine):

    communication_conf = CommunicationConfig()

    try:
        print('host={} | port={} | action={}'.format(vm.instance_ip, communication_conf.socket_port, Daemon.TEST))
        communicator = Communicator(host=vm.instance_ip, port=communication_conf.socket_port)
        print("Created communicator")
        communicator.send(action=Daemon.TEST, value={'task_id': None, 'command': None})

        if communicator.response['status'] == 'success':
            return True

    except Exception as e:
        logging.error(e)
        return False


def __execution_loop(vm: VirtualMachine, task: Task):

    # Start the VM in the cloud
    status = vm.deploy()

    logging.info("<Executor {}-{}>: Instance-id {} - Status {}".format(task.task_id, vm.instance_id,
                                                                       vm.instance_id, status))
    if status:
        try:
            vm.prepare_vm()
        except Exception as e:
            logging.error(e)

    try:
        __prepare_daemon(vm)
    except Exception as e:
        logging.error(e)

    # indicate that the VM is ready to execute
    vm.ready = True

    # start Execution when instance is RUNNING
    if vm.state == CloudManager.RUNNING:

        # create a executor and start task
        executor = Executor(
            task=task,
            vm=vm
        )
        # start the executor loop to execute the task
        executor.thread.start()

    # if vm.state == CloudManager.RUNNING:

        # while self.debug_wait_command:
        #     time.sleep(5)

        # status = vm.terminate()

        # if status:
        #     logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))

    # else:
        # Error to start VM
        # logging.error("<Dispatcher> Instance type: {} Was not started".format(vm.instance_type.type))


def test_dispatcher_with_daemon():
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-09685b54c80020d8c',
        ebs_device_name='/dev/xvdf',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031}
    )

    task = Task(
        task_id=2,
        command="ls",
        runtime={'t2.micro': 100},
        generic_ckpt=False
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand'
    )

    vm.instance_id = 'i-0dd21b2167699e7dd'

    __prepare_logging()

    __execution_loop(vm=vm, task=task)
