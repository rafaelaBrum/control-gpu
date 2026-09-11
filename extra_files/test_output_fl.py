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
    parser.add_argument('--job_file', help="job file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    parser.add_argument('--loc_file', help="loc file name", type=str, default=None)
    parser.add_argument('--pre_file', help="pre scheduling file name", type=str, default=None)
    parser.add_argument('--input_file', help="input metrics file name", type=str, default=None)
    parser.add_argument('--scheduler_name', help="Scheduler name", type=str, default=None)
    parser.add_argument('--map_file', help="map file name", type=str, default=None)
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

    parser.add_argument('--strategy', help="Strategy to use in server aggregation", type=str, default=None,
                        required=False)
    parser.add_argument('--frequency_ckpt', help='Frequency of checkpointing when using FedAvgSave', type=int,
                        default=None)

    parser.add_argument('--num_seed', help="Seed to be used by the clients to randomly shuffle their dataset",
                        default=None, required=False)

    parser.add_argument('--emulated', action='store_true', dest='emulated',
                        help='Using CloudLab to execute experiments', default=False)

    parser.add_argument('volume_id', help="Volume id to be attached", type=str)
    parser.add_argument('cloud_provider', help="Provider which the volume are (AWS or GCP). Default: AWS", type=str,
                        default='aws')
    parser.add_argument('cloud_zone', help="Zone in which the volume are", type=str, default='')

    args = parser.parse_args()
    loader = Loader(args=args)
    volume_id = args.volume_id

    if args.cloud_provider in (CloudManager.EC2, CloudManager.AWS):
        instance = InstanceType(
            provider=CloudManager.EC2,
            instance_type='t2.micro',
            image_id='ami-0e2423601b8f14e2b',
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
        loc_id = 'AWS_' + args.cloud_zone[:-1]
        region = loader.loc[loc_id]
    elif args.cloud_provider in (CloudManager.GCLOUD, CloudManager.GCP):
        instance = InstanceType(
            provider=CloudManager.GCLOUD,
            instance_type='e2-micro',
            image_id='disk-ubuntu-flower-server',
            restrictions={'on-demand': 1,
                          'preemptible': 1},
            prices={'on-demand': 0.001,
                    'preemptible': 0.000031},
            vcpu=2,
            ebs_device_name='/dev/sdb',
            gpu='no',
            count_gpu=0,
            memory=4,
            locations=''
        )
        loc_id = 'GCP_' + args.cloud_zone[:-2]
        region = loader.loc[loc_id]
    else:
        logging.error("Cloud provider not available ({})".format(args.cloud_provider))
        return

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand',
        loader=loader,
        zone=args.cloud_zone
    )

    __prepare_logging()

    if volume_id is not None:
        vm.volume_id = volume_id
        vm.disk_name = volume_id
    vm.region = region

    vm.deploy(type_task='server', zone=args.cloud_zone)

    vm.prepare_vm('server')


if __name__ == "__main__":
    main()
