import argparse
import os
import time

from control.domain.instance_type import InstanceType

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

from control.config.logging_config import LoggingConfig
from control.util.loader import Loader

from control.util.ssh_client import SSHClient

import logging

from pathlib import Path

from typing import List

import threading


def main():
    parser = argparse.ArgumentParser(description='Control GPU - v. 0.0.1')

    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--n_parties', type=int, required=True)

    parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
    parser.add_argument('--task_file', help="task file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)

    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)

    parser.add_argument('--command', default='control')

    parser.add_argument('--instance_type', required=True)

    parser.add_argument('--rounds', default=10)

    args = parser.parse_args()

    loader = Loader(args=args)

    __prepare_logging()

    n_parties = args.n_parties

    vm_server = create_server_on_demand(loader, n_parties, args.rounds)

    instance_type = args.instance_type

    logging.info("Server created!")

    # time.sleep(10)

    threads: List[threading.Thread] = []

    for i in range(n_parties):
        x = threading.Thread(target=controlling_client_flower, args=(loader,
                                                                     vm_server.instance_private_ip,
                                                                     i, os.path.join(args.folder, 'client'),
                                                                     instance_type, n_parties))
        threads.append(x)
        x.start()
        time.sleep(5)

    while vm_server.state not in (CloudManager.TERMINATED, None):
        # print("Testing the server")
        if has_command_finished(vm_server):
            logging.info('Server has finished execution!')
            finish_vm(vm_server, os.path.join(args.folder, 'server'), 'screen_log')
        time.sleep(5)

    cost = vm_server.uptime.seconds * (vm_server.price / 3600.0)  # price in seconds'

    logging.info("Server cost {}".format(cost))

    for i in range(n_parties):
        threads[i].join()


# Global functions
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
        cmd3 = 'sudo chown ubuntu:ubuntu -R {}'.format(path)
        cmd4 = 'chmod 775 -R {}'.format(path)

        logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd3))
        vm.ssh.execute_command(cmd3, output=True)

        logging.info("<VirtualMachine {}>: - {} ".format(vm.instance_id, cmd4))
        vm.ssh.execute_command(cmd4, output=True)


def __create_s3(vm: VirtualMachine, path):

    logging.info("<VirtualMachine {}>: - Mounting S3FS".format(vm.instance_id))

    # prepare S3FS
    cmd1 = 'echo {}:{} > $HOME/.passwd-s3fs'.format(vm.manager.credentials.access_key,
                                                    vm.manager.credentials.secret_key)

    cmd2 = 'sudo chmod 600 $HOME/.passwd-s3fs'

    # Mount the bucket
    cmd3 = 'sudo s3fs {} ' \
           '-o use_cache=/tmp -o allow_other -o uid={} -o gid={} ' \
           '-o mp_umask=002 -o multireq_max=5 {}'.format(vm.manager.s3_conf.bucket_name,
                                                         vm.manager.s3_conf.vm_uid,
                                                         vm.manager.s3_conf.vm_gid,
                                                         path)

    logging.info("<VirtualMachine {}>: - Creating .passwd-s3fs".format(vm.instance_id))
    vm.ssh.execute_command(cmd1, output=True)

    logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd2))
    vm.ssh.execute_command(cmd2, output=True)

    logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd3))
    vm.ssh.execute_command(cmd3, output=True)


def __call_control():
    print("aqui")


def __send_zip_file(vm: VirtualMachine, file):
    vm.ssh.put_file(source=vm.loader.application_conf.flower_path,
                    target=vm.loader.ec2_conf.home_path,
                    item=file)

    cmd1 = f'unzip {file} -d .'

    stdout, stderr, code_return = vm.ssh.execute_command(cmd1, output=True)
    logging.info("<VirtualMachine {}>: output of '{}' is '{}'".format(vm.instance_id, cmd1, stdout))
    logging.info("<VirtualMachine {}>: error output of '{}' is '{}'".format(vm.instance_id, cmd1, stderr))


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


def finish_vm(vm: VirtualMachine, folder, item_name):
    try:
        os.makedirs(os.path.join(vm.loader.communication_conf.key_path, 'control-gpu', 'logs', folder))
    except Exception as _:
        pass
    try:
        vm.ssh.get_file(source=vm.loader.ec2_conf.home_path,
                        target=Path(vm.loader.communication_conf.key_path, 'control-gpu', 'logs', folder),
                        item=item_name)
    except Exception as e:
        logging.error("<VirtualMachine {}>:: SSH CONNECTION ERROR - {}".format(vm.instance_id, e))

    status = vm.terminate()

    if status:
        logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


def has_command_finished(vm: VirtualMachine):
    cmd = "screen -list | grep test"

    stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)

    if 'test' in stdout:
        finished = False
    else:
        finished = True
    return finished


# Server functions
def __prepare_vm_server(vm: VirtualMachine, n_parties, rounds):
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
            #
            # __send_zip_file(vm, vm.loader.application_conf.server_flower_file)
            vm.ssh.put_file(source=vm.loader.application_conf.flower_path,
                            target=vm.loader.ec2_conf.home_path,
                            item="fedavg_strategy.py")

            vm.ssh.put_file(source=vm.loader.application_conf.flower_path,
                            target=vm.loader.ec2_conf.home_path,
                            item=vm.loader.application_conf.server_flower_file.replace('.zip', '.py'))

            # Start Daemon
            logging.info("<VirtualMachine {}>: - Starting Server".format(vm.instance_id))

            # cmd_daemon = "ls tests"
            cmd_daemon = "python3 {0} --rounds {1} --sample_fraction 1 --min_sample_size {2} " \
                         "--min_num_clients {2}".format(os.path.join(vm.loader.ec2_conf.home_path,
                                                                     vm.loader.application_conf.
                                                                     server_flower_file.
                                                                     replace('.zip', '.py')),
                                                        rounds,
                                                        n_parties)

            cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{}"'.format(cmd_daemon)
            # cmd_screen = '{}'.format(cmd_daemon)

            logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd_screen))

            stdout, stderr, code_return = vm.ssh.execute_command(cmd_screen, output=True)
            print(stdout)

            # vm.deploy_overhead = datetime.now() - vm.start_deploy

        else:

            logging.error("<VirtualMachine {}>:: SSH CONNECTION ERROR".format(vm.instance_id))
            raise Exception("<VirtualMachine {}>:: SSH Exception ERROR".format(vm.instance_id))


def create_server_on_demand(loader: Loader, n_parties, n_rounds):
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.xlarge',
        image_id='ami-03e15d31e4fab2356',
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

    vm.deploy()

    __prepare_vm_server(vm, n_parties, n_rounds)

    return vm


# Client functions
def __prepare_vm_client(vm: VirtualMachine, server_ip, client_id, train_folder, test_folder):
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
            vm.ssh.execute_command('mkdir -p {}'.format(vm.loader.file_system_conf.path_ebs), output=True)

            __create_s3(vm, vm.loader.file_system_conf.path)

            __create_ebs(vm, vm.loader.file_system_conf.path_ebs)

            # keep ssh live
            # vm.ssh.execute_command("$HOME/.ssh/config")

            # cmd = 'mkdir {}_{}/'.format(vm.loader.cudalign_task.task_id, vm.loader.execution_id)
            #
            # logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd))

            # stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)
            # print(stdout)

            # Send daemon file

            __send_zip_file(vm, vm.loader.application_conf.client_flower_file)

            # Start Daemon
            logging.info("<VirtualMachine {}>: - Starting Client".format(vm.instance_id))

            cmd0 = 'mkdir {}'.format(os.path.join(vm.loader.file_system_conf.path_ebs, 'logs'))
            cmd1 = 'mkdir {}'.format(os.path.join(vm.loader.file_system_conf.path_ebs, 'results'))

            stdout, stderr, code_return = vm.ssh.execute_command(cmd0, output=True)
            print(stdout)
            stdout, stderr, code_return = vm.ssh.execute_command(cmd1, output=True)
            print(stdout)

            # cmd_daemon = "ls tests"
            cmd_daemon = "python3 {0} -i -v --train -predst {1} -split 0.9 0.1 0.0 -d -b 32 -tn " \
                         "-out {2} -cpu 4 -gpu 0 -wpath {3} -model_dir {3} -logdir {3} " \
                         "-server_address {4} -tdim 240 240 -f1 10 " \
                         "-cache {3} -test_dir {5} ".format(os.path.join(vm.loader.ec2_conf.home_path,
                                                                         vm.loader.application_conf.client_flower_file.
                                                                         replace('.zip', '.py')),
                                                            os.path.join(vm.loader.ec2_conf.input_path, train_folder),
                                                            os.path.join(vm.loader.file_system_conf.path_ebs, 'logs'),
                                                            os.path.join(vm.loader.file_system_conf.path_ebs,
                                                                         'results'),
                                                            server_ip,
                                                            os.path.join(vm.loader.ec2_conf.input_path, test_folder))

            cmd_screen = 'screen -L -Logfile $HOME/screen_log_{} -S test -dm bash -c "{}"'.format(client_id, cmd_daemon)
            # cmd_screen = '{}'.format(cmd_daemon)

            logging.info("<VirtualMachine {}>: - {}".format(vm.instance_id, cmd_screen))

            stdout, stderr, code_return = vm.ssh.execute_command(cmd_screen, output=True)
            print(stdout)

            # vm.deploy_overhead = datetime.now() - vm.start_deploy

        else:

            logging.error("<VirtualMachine {}>:: SSH CONNECTION ERROR".format(vm.instance_id))
            raise Exception("<VirtualMachine {}>:: SSH Exception ERROR".format(vm.instance_id))


def create_client_on_demand(loader: Loader, server_ip, client_id, instance_type, n_parties):
    if instance_type == 'g4dn.xlarge':
        instance = InstanceType(
            provider=CloudManager.EC2,
            instance_type='g4dn.2xlarge',
            image_id='ami-080af420cdfb56e39',
            ebs_device_name='/dev/nvme2n1',
            restrictions={'on-demand': 1,
                          'preemptible': 1},
            prices={'on-demand': 0.001,
                    'preemptible': 0.000031}
        )
    elif instance_type == 'c5d.2xlarge':
        instance = InstanceType(
            provider=CloudManager.EC2,
            instance_type='c5d.2xlarge',
            image_id='ami-080af420cdfb56e39',
            ebs_device_name='/dev/nvme2n1',
            restrictions={'on-demand': 1,
                          'preemptible': 1},
            prices={'on-demand': 0.001,
                    'preemptible': 0.000031}
        )
    elif instance_type == 'r5dn.xlarge':
        instance = InstanceType(
            provider=CloudManager.EC2,
            instance_type='r5dn.xlarge',
            image_id='ami-080af420cdfb56e39',
            ebs_device_name='/dev/nvme2n1',
            restrictions={'on-demand': 1,
                          'preemptible': 1},
            prices={'on-demand': 0.001,
                    'preemptible': 0.000031}
        )
    elif instance_type == 'd3.xlarge':
        instance = InstanceType(
            provider=CloudManager.EC2,
            instance_type='d3.xlarge',
            image_id='ami-080af420cdfb56e39',
            ebs_device_name='/dev/nvme4n1',
            restrictions={'on-demand': 1,
                          'preemptible': 1},
            prices={'on-demand': 0.001,
                    'preemptible': 0.000031}
        )
    else:
        return

    vm = VirtualMachine(
        instance_type=instance,
        market='preemptible',
        loader=loader,
    )

    vm.deploy()

    train_folder = f'data/CellRep/{n_parties}_clients/{client_id}/trainset'
    test_folder = f'data/CellRep/{n_parties}_clients/{client_id}/testset'

    __prepare_vm_client(vm, server_ip, client_id, train_folder, test_folder)

    return vm


def controlling_client_flower(loader: Loader, server_ip: str,
                              client_id: int, folder_log: str, instance_type: str, n_parties: int):
    vm = create_client_on_demand(loader, f"{server_ip}:8080", client_id, instance_type, n_parties)
    logging.info(f"Client_{client_id} created!")

    # input("waiting...")

    while vm.state not in (CloudManager.TERMINATED, None):
        # print(f"Testing client_{client_id}")
        if has_command_finished(vm):
            logging.info(f'Client {client_id} has finished execution!')
            finish_vm(vm, folder_log, f'screen_log_{client_id}')
        time.sleep(5)

    cost = vm.uptime.seconds * (vm.price / 3600.0)  # price in seconds'

    logging.info("Client {} cost {}".format(client_id, cost))


if __name__ == "__main__":
    main()
