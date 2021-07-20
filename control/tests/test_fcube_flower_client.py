import argparse
import os

from control.domain.instance_type import InstanceType

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

from control.config.logging_config import LoggingConfig
from control.util.loader import Loader

from control.util.ssh_client import SSHClient

import logging

from pathlib import Path


def __create_ebs(vm, path):
    internal_device_name = vm.instance_type.ebs_device_name

    logging.info("<VirtualMachine {}>: - Mounting EBS".format(vm.instance_id))

    if vm.create_file_system:
        cmd1 = 'sudo mkfs.ext4 {}'.format(internal_device_name)
        logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd1))
        vm.ssh.execute_command(cmd1, output=True)

    # Mount Directory
    cmd2 = 'sudo mount {} {}'.format(internal_device_name, path)
    logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd2))

    vm.ssh.execute_command(cmd2, output=True)  # mount directory

    if vm.create_file_system:
        cmd3 = 'sudo chown ubuntu:ubuntu -R {}'.format(vm.loader.file_system_conf.path)
        cmd4 = 'chmod 775 -R {}'.format(vm.loader.file_system_conf.path)

        logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd3))
        vm.ssh.execute_command(cmd3, output=True)

        logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd4))
        vm.ssh.execute_command(cmd4, output=True)


def __prepare_vm_client(vm: VirtualMachine, server_ip):
    if not vm.failed_to_created:

        # update instance IP
        vm.update_ip()
        # Start a new SSH Client
        vm.ssh = SSHClient(vm.instance_public_ip)

        # try to open the connection
        if vm.ssh.open_connection():

            logging.info("<VirtualMachine {}>: - Creating directory {}".format(vm.instance_id,
                                                                               vm.loader.file_system_conf.path))
            # create directory
            vm.ssh.execute_command('mkdir -p {}'.format(vm.loader.file_system_conf.path), output=True)

            __create_ebs(vm, vm.loader.file_system_conf.path)

            # keep ssh live
            # vm.ssh.execute_command("$HOME/.ssh/config")

            # cmd = 'mkdir {}_{}/'.format(vm.loader.cudalign_task.task_id, vm.loader.execution_id)
            #
            # logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd))

            # stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)
            # print(stdout)

            # Send daemon file
            vm.ssh.put_file(source=vm.loader.application_conf.flower_path,
                            target=vm.loader.ec2_conf.home_path,
                            item=vm.loader.application_conf.client_flower_file)


            # Send dataset files
            vm.ssh.put_file(source=Path(vm.loader.application_conf.data_path, vm.loader.application_conf.dataset, '0'),
                            target=vm.loader.ec2_conf.input_path,
                            item='X_train.csv')
            vm.ssh.put_file(source=Path(vm.loader.application_conf.data_path, vm.loader.application_conf.dataset, '0'),
                            target=vm.loader.ec2_conf.input_path,
                            item='X_test.csv')
            vm.ssh.put_file(source=Path(vm.loader.application_conf.data_path, vm.loader.application_conf.dataset, '0'),
                            target=vm.loader.ec2_conf.input_path,
                            item='y_train.csv')
            vm.ssh.put_file(source=Path(vm.loader.application_conf.data_path, vm.loader.application_conf.dataset, '0'),
                            target=vm.loader.ec2_conf.input_path,
                            item='y_test.csv')

            # create execution folder
            # vm.root_folder = os.path.join(vm.loader.file_system_conf.path,
            #                                 '{}_{}'.format(vm.loader.cudalign_task.task_id,
            #                                                vm.loader.execution_id))
            #
            # vm.ssh.execute_command('mkdir -p {}'.format(vm.root_folder), output=True)

            # Start Daemon
            logging.info("<VirtualMachine {}>: - Starting Client".format(vm.instance_id))

            # cmd_daemon = "ls tests"
            cmd_daemon = "python3 {} " \
                         "--server_address {} " \
                         "--path_dataset {} " \
                         "--batch-size {} ".format(os.path.join(vm.loader.ec2_conf.home_path,
                                                              vm.loader.application_conf.client_flower_file),
                                                 server_ip,
                                                 vm.loader.ec2_conf.input_path,
                                                 64)

            cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{}"'.format(cmd_daemon)
            # cmd_screen = '{}'.format(cmd_daemon)

            logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd_screen))

            stdout, stderr, code_return = vm.ssh.execute_command(cmd_screen, output=True)
            print(stdout)

            # vm.deploy_overhead = datetime.now() - vm.start_deploy

        else:

            logging.error("<VirtualMachine {}>:: SSH CONNECTION ERROR".format(vm.instance_id))
            raise Exception("<VirtualMachine {}>:: SSH Exception ERROR".format(vm.instance_id))


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


def test_client_on_demand(loader: Loader, server_ip):
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.micro',
        image_id='ami-0419c40eab85fd7e9',
        ebs_device_name='/dev/xvdf',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031}
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand',
        loader=loader,
    )

    __prepare_logging()

    vm.deploy()

    __prepare_vm_client(vm, server_ip)

    print(vm.instance_public_ip)

    # var = 'n'
    #
    # while var.lower() != 'y':
    #     var = input("OK to finish instance? [y/n]")

    cmd = "screen -list | grep test"

    stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)

    while 'test' in stdout:
        stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)
    print("Client has finished!")

    vm.ssh.get_file(source=vm.loader.ec2_conf.home_path,
                    target=Path(vm.loader.communication_conf.key_path, 'control-gpu', 'logs', 'client'),
                    item='screen_log')

    status = vm.terminate()

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


def main():
    parser = argparse.ArgumentParser(description='Control GPU - v. 0.0.1')

    parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
    parser.add_argument('--task_file', help="task file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    # parser.add_argument('--map_file', help="map file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
    # parser.add_argument('--ac_size_seconds', help="Define the size of the Logical Allocation Cycle (seconds)",
    #                     type=int, default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)

    # parser.add_argument('--resume_rate', help="Resume rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)",
    #                     type=float, default=None)
    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)

    # parser.add_argument('--scheduler_name',
    #                     help="Scheduler name - Currently supported Schedulers are: " + ", ".join(
    #                         Scheduler.scheduler_names),
    #                     type=str, default=None)

    # parser.add_argument('--notify', help='Send an email to notify the end of the execution (control mode)',
    #                     action='store_true', default=False)

    # options_map = {
    #     'control': __call_control,
    #     # 'map': __call_primary_scheduling,
    #     # 'recreate_db': __call_recreate_database,
    #     # 'info': __print_execution_info,
    # }
    parser.add_argument('--command', default='control')

    loader = Loader(args=parser.parse_args())

    # server_ip = input("Enter server ip and port:")
    server_ip = "172.31.47.85:8080"

    test_client_on_demand(loader, server_ip)


def __call_control():
    print("aqui")
