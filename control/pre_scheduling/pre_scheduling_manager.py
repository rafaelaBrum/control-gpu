import json
import logging
import os
import threading
import time
from typing import Dict, List
from copy import deepcopy

from control.domain.instance_type import InstanceType
from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine
from control.util.loader import Loader
from control.util.ssh_client import SSHClient

from control.domain.app_specific.fl_client_task import FLClientTask

# TODO: remove this from pre_scheduling
aws_g5_ami_east = 'ami-04a48a1810b5f47b3'

aws_g5_ami_west = 'ami-00c4927f4e41cef7c'

gcp_a2_ami = 'disk-ubuntu-flower-client-a2-instances'

instance_aws = InstanceType(
    provider=CloudManager.EC2,
    instance_type='t2.micro',
    image_id='ami-0c20212ac4ce26a60',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    ebs_device_name='/dev/xvdf',
    gpu='no',
    count_gpu=0,
    vcpu='2',
    memory=0
)

instance_aws_rpc = InstanceType(
    provider=CloudManager.EC2,
    instance_type='t2.xlarge',
    image_id='ami-0c20212ac4ce26a60',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    ebs_device_name='/dev/xvdf',
    gpu='no',
    count_gpu=0,
    vcpu=2,
    memory=8
)

instance_aws_rpc_concurrent_server = InstanceType(
    provider=CloudManager.EC2,
    instance_type='t2.xlarge',
    image_id='ami-0c20212ac4ce26a60',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    ebs_device_name='/dev/xvdf',
    gpu='no',
    count_gpu=0,
    vcpu=2,
    memory=8
)

instance_gcp = InstanceType(
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
    memory=4
)

instance_gcp_rpc = InstanceType(
    provider=CloudManager.GCLOUD,
    instance_type='e2-standard-4',
    image_id='disk-ubuntu-flower-server',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    vcpu=4,
    ebs_device_name='/dev/sdb',
    gpu='no',
    count_gpu=0,
    memory=16
)


def has_command_finished(vm):
    cmd = "screen -list | grep test"

    stdout, stderr, code_return = vm.ssh.execute_command(cmd, output=True)

    if 'test' in stdout:
        finished = False
    else:
        finished = True
    return finished


class PreSchedulingManager:
    def __init__(self, loader: Loader):

        self.loader = loader
        self.rtt_values: Dict[str, Dict[str, float]] = {}
        self.exec_times: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.rpc_times: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.rpc_times_concurrent: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]] = {}

        self.stop_execution = self.__read_json()

    def calculate_rtt_values(self):
        logging.info("<PreSchedulerManager>: Calculating RTT values")
        loc_copy = deepcopy(self.loader.loc)
        for region_id, region in self.loader.loc.items():
            if region.provider in (CloudManager.EC2, CloudManager.AWS):
                vm_initial = VirtualMachine(instance_type=instance_aws, market='on-demand', loader=self.loader)
            elif region.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                vm_initial = VirtualMachine(instance_type=instance_gcp, market='on-demand', loader=self.loader)
            else:
                logging.error(f"<PreSchedulingManager>: {region.provider} does not have support ({region_id})")
                return
            key_file_initial = ''
            if region.key_file != '':
                key_file_initial = region.key_file.split('.')[0]
            for zone in region.zones:
                id_rtt = region_id + '_' + zone
                logging.info(f"<PreSchedulerManager>: Initialing zone {zone} of region {region.region}"
                             f" or provider {region.provider}")
                vm_initial.instance_type.image_id = region.server_image_id
                vm_initial.region = region
                vm_initial.zone = zone
                if id_rtt not in self.rtt_values:
                    self.rtt_values[id_rtt] = {}
                for region_copy in loc_copy.values():
                    if region_copy.provider in (CloudManager.EC2, CloudManager.AWS):
                        vm_final = VirtualMachine(instance_type=instance_aws, market='on-demand', loader=self.loader)
                    elif region_copy.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                        vm_final = VirtualMachine(instance_type=instance_gcp, market='on-demand', loader=self.loader)
                    else:
                        logging.error(
                            f"<PreSchedulingManager>: {region_copy.provider} does not have support ({region_copy.id})")
                        return
                    vm_final.instance_type.image_id = region_copy.server_image_id
                    vm_final.region = region_copy
                    key_file = ''
                    if region_copy.key_file != '':
                        key_file = region_copy.key_file.split('.')[0]
                    for zone_copy in region_copy.zones:
                        id_rtt_final = region_copy.id + '_' + zone_copy
                        if id_rtt_final in self.rtt_values[id_rtt]:
                            continue
                        logging.info(f"<PreSchedulerManager>: Testing with zone {zone_copy} of "
                                     f"region {region_copy.region} of provider {region_copy.provider}")
                        if not vm_initial.failed_to_created:
                            vm_initial.deploy(zone=zone, needs_volume=False, key_name=key_file_initial,
                                              type_task='server')
                            if not vm_initial.failed_to_created:
                                vm_initial.update_ip(zone=zone)
                        vm_final.zone = zone_copy
                        vm_final.deploy(zone=zone_copy, needs_volume=False, key_name=key_file, type_task='server')
                        if not vm_final.failed_to_created:
                            # update instance IP
                            vm_final.update_ip(zone=zone_copy)
                            self.rtt_values[id_rtt][id_rtt_final] = self.__exec_rtt_vms(vm_initial, vm_final,
                                                                                        region.key_file,
                                                                                        region_copy.key_file)
                            status = vm_final.terminate(wait=False, zone=zone_copy)
                            if status:
                                vm_final.instance_id = None
                                vm_final.failed_to_created = False
                        else:
                            vm_final.instance_id = None
                            vm_final.failed_to_created = False
                loc_copy[region_id].zones.remove(zone)
                if not vm_initial.failed_to_created:
                    status = vm_initial.terminate(wait=False, zone=zone)
                    if status:
                        vm_initial.instance_id = None
                        vm_initial.failed_to_created = False
                else:
                    vm_initial.instance_id = None
                    vm_initial.failed_to_created = False
            loc_copy.pop(region_id, None)
        # print("rtt_values")
        # print(json.dumps(self.rtt_values, sort_keys=True, indent=4))

    def __exec_rtt_vms(self, vm_initial, vm_final, key_initial, key_final):

        if key_initial == '':
            if vm_initial.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                key_initial = self.loader.ec2_conf.key_file
            elif vm_initial.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                key_initial = self.loader.gcp_conf.key_file
            else:
                logging.error(f"PreSchedulerManager>: "
                              f"{vm_initial.instance_type.provider} does not have support")

        if key_final == '':
            if vm_final.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                key_final = self.loader.ec2_conf.key_file
            elif vm_final.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                key_final = self.loader.gcp_conf.key_file
            else:
                logging.error(f"PreSchedulerManager>: "
                              f"{vm_final.instance_type.provider} does not have support")
        # Start a new SSH Client
        if vm_final.instance_type.provider == CloudManager.EC2:
            vm_final.ssh = SSHClient(vm_final.instance_public_ip, self.loader.ec2_conf.key_path,
                                     key_final, self.loader.ec2_conf.vm_user)
        elif vm_final.instance_type.provider == CloudManager.GCLOUD:
            vm_final.ssh = SSHClient(vm_final.instance_public_ip, self.loader.gcp_conf.key_path,
                                     key_final, self.loader.gcp_conf.vm_user)

        # try to open the connection
        if vm_final.ssh.open_connection():

            logging.info("<VirtualMachine {}>: - Sending RTT test".format(vm_final.instance_id,
                                                                          self.loader.file_system_conf.path))

            item = self.loader.pre_sched_conf.rtt_file

            if vm_initial.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                key_path = self.loader.ec2_conf.key_path
                item_key = key_initial
                vm_user = self.loader.ec2_conf.vm_user
            elif vm_initial.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                key_path = self.loader.gcp_conf.key_path
                item_key = key_initial
                vm_user = self.loader.gcp_conf.vm_user
            else:
                logging.error(f"PreSchedulerManager>: "
                              f"{vm_initial.instance_type.provider} does not have support")
                return

            # Send files
            if vm_final.instance_type.provider == CloudManager.EC2:

                vm_final.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                      target=self.loader.ec2_conf.home_path,
                                      item=item)

                vm_final.ssh.put_file(source=key_path,
                                      target=self.loader.ec2_conf.home_path,
                                      item=item_key)

                cmd_daemon = "python3.7 {} " \
                             "--ip {} " \
                             "--path {} " \
                             "--file {} " \
                             "--user {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                              self.loader.pre_sched_conf.rtt_file),
                                                 vm_initial.instance_public_ip,
                                                 self.loader.ec2_conf.home_path,
                                                 item_key,
                                                 vm_user)

            elif vm_final.instance_type.provider == CloudManager.GCLOUD:

                vm_final.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                      target=self.loader.gcp_conf.home_path,
                                      item=item)

                vm_final.ssh.put_file(source=key_path,
                                      target=self.loader.gcp_conf.home_path,
                                      item=item_key)

                cmd_daemon = "python3.7 {} " \
                             "--ip {} " \
                             "--path {} " \
                             "--file {} " \
                             "--user {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                              item),
                                                 vm_initial.instance_public_ip,
                                                 self.loader.gcp_conf.home_path,
                                                 item_key,
                                                 vm_user)
            else:
                cmd_daemon = ""

            cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
            logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                         vm_final.instance_id))

            stdout, stderr, code_return = vm_final.ssh.execute_command(cmd_screen, output=True)
            print(stdout)

            while not has_command_finished(vm_final):
                continue

            cmd = "cat $HOME/screen_log"
            logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd,
                                                                                         vm_final.instance_id))

            trying = 0
            rtt_value = -1
            while trying < 3:
                try:
                    stdout, stderr, code_return = vm_final.ssh.execute_command(cmd, output=True)

                    rtt_value = float(stdout.split(" ")[-1])
                    break
                except Exception as e:
                    print("Error to convert: '", stdout, "'")
                    print(f"Try {trying}/3")
                    print(e)
                    trying = trying + 1
                    time.sleep(5)

            return rtt_value

        else:

            logging.error("<PreScheduler> VirtualMachine {}:: SSH CONNECTION ERROR".format(vm_final.instance_id))
            raise Exception("<PreScheduler> VirtualMachine {}:: SSH Exception ERROR".format(vm_final.instance_id))

    def get_first_rounds_times(self):
        try:
            logging.info("<PreSchedulerManager>: Computing training times")
            clients = self.loader.job.client_tasks
            env_aws, env_gcp = self.separate_env_per_cloud()
            loc_aws, loc_gcp = self.separate_loc_per_cloud()
            logging.info("<PreSchedulerManager>: Calculating AWS training times")
            for env_id, env in env_aws.items():
                if not env.have_gpu:
                    continue
                logging.info(f"<PreSchedulerManager>: Using instance {env_id}")
                if env_id not in self.exec_times:
                    self.exec_times[env_id] = {}
                vm = VirtualMachine(instance_type=env, market='on-demand', loader=self.loader)
                for loc_id, region in loc_aws.items():
                    if loc_id in self.exec_times[env_id]:
                        skip_loc = True
                        for cli in clients.values():
                            if str(cli.client_id) not in self.exec_times[env_id][loc_id]:
                                skip_loc = False
                        if skip_loc:
                            continue
                    else:
                        self.exec_times[env_id][loc_id] = {}
                    logging.info(f"<PreSchedulerManager>: Testing in region {region.region}")
                    key_name = region.key_file.split('.')[0]
                    vm.instance_type.image_id = region.client_image_id
                    new_ami = ''
                    if "g5" in vm.instance_type.type:
                        if region.region == "us-east-1":
                            new_ami = aws_g5_ami_east
                        elif region.region == "us-west-2":
                            new_ami = aws_g5_ami_west
                    elif "a2" in vm.instance_type.type:
                        new_ami = gcp_a2_ami
                    vm.region = region
                    key_file = region.key_file
                    final_zone = ''
                    for zone in region.zones:
                        try:
                            vm.zone = zone
                            vm.deploy(zone=zone, needs_volume=False,
                                      key_name=key_name, type_task='client', ami_id=new_ami)
                            final_zone = zone
                            break
                        except Exception as e:
                            logging.error(f'<PreSchedulerManager>: Error with zone {zone}')
                            logging.error(e)
                            vm.instance_id = None
                    if not vm.failed_to_created:
                        # update instance IP
                        vm.update_ip(zone=final_zone)
                        for cli in clients.values():
                            if str(cli.client_id) in self.exec_times[env_id][loc_id]:
                                continue
                            logging.info(f"<PreSchedulerManager>: Testing client {cli.client_id} "
                                         f"in region {region.region}")
                            self.exec_times[env_id][loc_id][str(cli.client_id)] = \
                                self.__compute_training_times(vm, key_file, cli)
                            vm.reboot()
                        print("final_zone", final_zone)
                        status = vm.terminate(wait=True, zone=final_zone)
                        if status:
                            vm.instance_id = None
                            vm.failed_to_created = False
                    else:
                        vm.instance_id = None
                        vm.failed_to_created = False
            logging.info("<PreSchedulerManager>: Calculating GCP training times")
            for env_id, env in env_gcp.items():
                if not env.have_gpu:
                    continue
                logging.info(f"<PreSchedulerManager>: Using instance {env_id}")
                if env_id not in self.exec_times:
                    self.exec_times[env_id] = {}
                vm = VirtualMachine(instance_type=env, market='on-demand', loader=self.loader)
                for loc_id, region in loc_gcp.items():
                    if loc_id in self.exec_times[env_id]:
                        skip_loc = True
                        for cli in clients.values():
                            if str(cli.client_id) not in self.exec_times[env_id][loc_id]:
                                skip_loc = False
                        if skip_loc:
                            continue
                    else:
                        self.exec_times[env_id][loc_id] = {}
                    logging.info(f"<PreSchedulerManager>: Testing in region {region.region}")
                    vm.instance_type.image_id = region.client_image_id
                    key_file = self.loader.gcp_conf.key_file
                    final_zone = ''
                    for zone in region.zones:
                        try:
                            status = vm.deploy(zone=zone, needs_volume=False, key_name=key_file, type_task='client')
                            final_zone = zone
                            if status:
                                break
                            else:
                                vm.instance_id = None
                        except Exception as e:
                            logging.error(f'<PreSchedulerManager>: Error with zone {zone}')
                            logging.error(e)
                            vm.instance_id = None
                    if not vm.failed_to_created:
                        # update instance IP
                        vm.update_ip(zone=final_zone)
                        for cli in clients.values():
                            if str(cli.client_id) in self.exec_times[env_id][loc_id]:
                                continue
                            logging.info(f"<PreSchedulerManager>: Testing client {cli.client_id} "
                                         f"in region {region.region}")
                            self.exec_times[env_id][loc_id][str(cli.client_id)] = \
                                self.__compute_training_times(vm, key_file, cli)
                            vm.reboot()
                        print("final_zone", final_zone)
                        status = vm.terminate(wait=True, zone=final_zone)
                        if status:
                            vm.instance_id = None
                            vm.failed_to_created = False
                    else:
                        vm.instance_id = None
                        vm.failed_to_created = False
        except Exception as e:
            logging.error(f'<PreSchedulerManager>: Error inside get_first_rounds_times')
            logging.error(e)

    def separate_env_per_cloud(self):
        env_aws = {}
        env_gcp = {}
        for env_id, env in self.loader.env.items():
            if env.provider in (CloudManager.EC2, CloudManager.AWS):
                env_aws[env_id] = env
            elif env.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                env_gcp[env_id] = env
            else:
                logging.error(f"<PreSchedulingManager>: {env.provider} does not have support ({env_id})")
        return env_aws, env_gcp

    def separate_loc_per_cloud(self):
        loc_aws = {}
        loc_gcp = {}
        for loc_id, loc in self.loader.loc.items():
            if loc.provider in (CloudManager.EC2, CloudManager.AWS):
                loc_aws[loc_id] = loc
            elif loc.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                loc_gcp[loc_id] = loc
            else:
                logging.error(f"<PreSchedulingManager>: {loc.provider} does not have support ({loc_id})")
        return loc_aws, loc_gcp

    def separate_cli_per_cloud(self):
        cli_aws = {}
        cli_gcp = {}
        for cli_id, cli in self.loader.job.client_tasks.items():
            if cli.bucket_provider in (CloudManager.EC2, CloudManager.AWS):
                cli_aws[cli_id] = cli
            elif cli.bucket_provider in (CloudManager.GCLOUD, CloudManager.GCP):
                cli_gcp[cli_id] = cli
            else:
                logging.error(f"<PreSchedulingManager>: {cli.bucket_provider} does not have support ({cli_id})")
        return cli_aws, cli_gcp

    def __compute_training_times(self, vm: VirtualMachine, key, cli: FLClientTask):

        if key == '':
            if vm.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                key = self.loader.ec2_conf.key_file
            elif vm.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                key = self.loader.gcp_conf.key_file
            else:
                logging.error(f"PreSchedulerManager>: "
                              f"{vm.instance_type.provider} does not have support")

        # Start a new SSH Client
        if vm.instance_type.provider == CloudManager.EC2:
            vm.ssh = SSHClient(vm.instance_public_ip, self.loader.ec2_conf.key_path,
                               key, self.loader.ec2_conf.vm_user)
        elif vm.instance_type.provider == CloudManager.GCLOUD:
            vm.ssh = SSHClient(vm.instance_public_ip, self.loader.gcp_conf.key_path,
                               key, self.loader.gcp_conf.vm_user)

        # try to open the connection
        if vm.ssh.open_connection():

            app_item = self.loader.pre_sched_conf.app_file
            train_item = self.loader.pre_sched_conf.train_file

            try:
                # # TODO: remove this from pre_scheduling
                # if "g5" in vm.instance_type.type:
                #     cmd_g5_1 = 'python3 -m pip uninstall tensorflow-gpu -y'
                #     logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_g5_1))
                #     stdout, stderr, code_return = vm.ssh.execute_command(cmd_g5_1, output=True)
                #     print(stdout)
                #     cmd_g5_2 = 'python3 -m pip install -r requirements_client_flower.txt'
                #     logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_g5_2))
                #     stdout, stderr, code_return = vm.ssh.execute_command(cmd_g5_2, output=True)
                #     print(stdout)

                logging.info("<VirtualMachine {}>: "
                             "- Creating directory {} ".format(vm.instance_id,
                                                               self.loader.file_system_conf.path_storage))
                # create directory
                vm.ssh.execute_command('mkdir -p {}'.format(self.loader.file_system_conf.path_storage),
                                       output=True)

                vm.create_bucket_pre_scheduling(self.loader.file_system_conf.path_storage, cli)

                logging.info(f"<VirtualMachine {vm.instance_id}>: - Sending Files Training test")

                # Send files
                if vm.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    vm.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                    target=self.loader.ec2_conf.home_path,
                                    item=app_item)

                    vm.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                    target=self.loader.ec2_conf.home_path,
                                    item=train_item)

                    cmd_before_daemon = "mkdir results/"

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_before_daemon))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd_before_daemon, output=True)
                    print(stdout)

                    cmd_remove = "rm results/*"

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_remove))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd_remove, output=True)
                    print(stdout)

                    cmd1 = f'unzip {app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd1))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "-i -v -predst {} " \
                                 "-split 0.9 0.10 0.00 " \
                                 "-net {} -data CellRep -d " \
                                 "-e {} -b {} -tdim 240 240 " \
                                 "-out logs/ -cpu {} -gpu {} " \
                                 "-tn -wpath results " \
                                 "-model_dir results " \
                                 "-logdir results " \
                                 "-cache results " \
                                 "-test_dir {} " \
                                 "-file {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                 self.loader.pre_sched_conf.train_file),
                                                    os.path.join(self.loader.file_system_conf.path_storage,
                                                                 cli.trainset_dir),
                                                    cli.net,
                                                    cli.train_epochs,
                                                    cli.batch,
                                                    vm.instance_type.vcpu,
                                                    vm.instance_type.count_gpu,
                                                    os.path.join(self.loader.file_system_conf.path_storage,
                                                                 cli.test_dir),
                                                    self.loader.pre_sched_conf.results_temp_file
                                                    )

                elif vm.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    vm.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                    target=self.loader.gcp_conf.home_path,
                                    item=app_item)

                    vm.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                    target=self.loader.gcp_conf.home_path,
                                    item=train_item)

                    cmd_before_daemon = "mkdir results/"

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_before_daemon))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd_before_daemon, output=True)
                    print(stdout)

                    cmd_remove = "rm results/*"

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_remove))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd_remove, output=True)
                    print(stdout)

                    cmd1 = f'unzip {app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd1))

                    stdout, stderr, code_return = vm.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "-i -v -predst {} " \
                                 "-split 0.9 0.10 0.00 " \
                                 "-net {} -data CellRep -d " \
                                 "-e {} -b {} -tdim 240 240 " \
                                 "-out logs/ -cpu {} -gpu {} " \
                                 "-tn -wpath results " \
                                 "-model_dir results " \
                                 "-logdir results " \
                                 "-cache results " \
                                 "-test_dir {} " \
                                 "-file {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                 self.loader.pre_sched_conf.train_file),
                                                    os.path.join(self.loader.file_system_conf.path_storage,
                                                                 cli.trainset_dir),
                                                    cli.net,
                                                    cli.train_epochs,
                                                    cli.batch,
                                                    vm.instance_type.vcpu,
                                                    vm.instance_type.count_gpu,
                                                    os.path.join(self.loader.file_system_conf.path_storage,
                                                                 cli.test_dir),
                                                    self.loader.pre_sched_conf.results_temp_file
                                                    )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm.instance_id))

                stdout, stderr, code_return = vm.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                # # TODO: remove this from pre_scheduling
                # if "g5" in vm.instance_type.type or "a2" in vm.instance_type.type:
                #     time.sleep(600)
                #     cmd_kill_screen = 'screen -X -S test quit'
                #     logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_kill_screen))
                #     stdout, stderr, code_return = vm.ssh.execute_command(cmd_kill_screen, output=True)
                #     print(stdout)
                #     cmd_daemon = "python3 {} " \
                #                  "-i -v -predst {} " \
                #                  "-split 0.9 0.10 0.00 " \
                #                  "-net {} -data CellRep -d " \
                #                  "-e {} -b {} -tdim 240 240 " \
                #                  "-out logs/ -cpu {} -gpu {} " \
                #                  "-tn -wpath results " \
                #                  "-model_dir results " \
                #                  "-logdir results " \
                #                  "-cache results " \
                #                  "-test_dir {} " \
                #                  "-file {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                #                                                  self.loader.pre_sched_conf.train_file),
                #                                     os.path.join(self.loader.file_system_conf.path_storage,
                #                                                  cli.trainset_dir),
                #                                     cli.net,
                #                                     cli.train_epochs,
                #                                     cli.batch,
                #                                     vm.instance_type.vcpu,
                #                                     vm.instance_type.count_gpu,
                #                                     os.path.join(self.loader.file_system_conf.path_storage,
                #                                                  cli.test_dir),
                #                                     self.loader.pre_sched_conf.results_temp_file
                #                                     )
                #     cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                #     logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                #                                                                                  vm.instance_id))
                #
                #     stdout, stderr, code_return = vm.ssh.execute_command(cmd_screen, output=True)
                #     print(stdout)

                while not has_command_finished(vm):
                    continue

                if vm.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                    vm.ssh.get_file(source=vm.loader.ec2_conf.home_path,
                                    target=self.loader.pre_sched_conf.path,
                                    item=self.loader.pre_sched_conf.results_temp_file)
                elif vm.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                    vm.ssh.get_file(source=vm.loader.gcp_conf.home_path,
                                    target=self.loader.pre_sched_conf.path,
                                    item=self.loader.pre_sched_conf.results_temp_file)

                vm.remove_bucket_pre_scheduling(self.loader.file_system_conf.path_storage, cli)

                cmd_remove = f"rm {app_item.split('.')[0]}* {self.loader.pre_sched_conf.results_temp_file} -r"

                logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_remove))

                stdout, stderr, code_return = vm.ssh.execute_command(cmd_remove, output=True)
                print(stdout)

                try:
                    with open(os.path.join(self.loader.pre_sched_conf.path,
                                           self.loader.pre_sched_conf.results_temp_file)) as f:
                        data = f.read()
                    times = json.loads(data)
                except Exception as e:
                    logging.error(e)
                    return {}

                return times
            except Exception as e:

                cmd_remove = f"rm {app_item.split('.')[0]}* {self.loader.pre_sched_conf.results_temp_file} -r"

                logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm.instance_id, cmd_remove))

                stdout, stderr, code_return = vm.ssh.execute_command(cmd_remove, output=True)
                print(stdout)

                logging.error(e)
                return {}
        else:

            logging.error("<PreScheduler> VirtualMachine {}:: SSH CONNECTION ERROR".format(vm.instance_id))
            raise Exception("<PreScheduler> VirtualMachine {}:: SSH Exception ERROR".format(vm.instance_id))

    def calculate_rpc_times(self):
        logging.info("<PreSchedulerManager>: Calculating RPC times")
        loc_copy = deepcopy(self.loader.loc)
        try:
            for region_id, region in self.loader.loc.items():
                if region.provider in (CloudManager.EC2, CloudManager.AWS):
                    vm_initial = VirtualMachine(instance_type=instance_aws_rpc, market='on-demand', loader=self.loader)
                elif region.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                    vm_initial = VirtualMachine(instance_type=instance_gcp_rpc, market='on-demand', loader=self.loader)
                else:
                    logging.error(f"<PreSchedulingManager>: {region.provider} does not have support ({region_id})")
                    return
                key_file_initial = ''
                if region.key_file != '':
                    key_file_initial = region.key_file.split('.')[0]
                for zone in region.zones:
                    id_rpc = region_id + '_' + zone
                    logging.info(f"<PreSchedulerManager>: Initialing zone {zone} of region {region.region}"
                                 f" or provider {region.provider}")
                    vm_initial.instance_type.image_id = region.server_image_id
                    vm_initial.region = region
                    vm_initial.zone = zone
                    if id_rpc not in self.rpc_times:
                        self.rpc_times[id_rpc] = {}
                    for region_copy in loc_copy.values():
                        if region_copy.provider in (CloudManager.EC2, CloudManager.AWS):
                            vm_final = VirtualMachine(instance_type=instance_aws_rpc, market='on-demand',
                                                      loader=self.loader)
                        elif region_copy.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                            vm_final = VirtualMachine(instance_type=instance_gcp_rpc, market='on-demand',
                                                      loader=self.loader)
                        else:
                            logging.error(
                                f"<PreSchedulingManager>: {region_copy.provider} does not have support ({region_copy.id})")
                            return
                        vm_final.instance_type.image_id = region_copy.server_image_id
                        vm_final.region = region_copy
                        key_file = ''
                        if region_copy.key_file != '':
                            key_file = region_copy.key_file.split('.')[0]
                        for zone_copy in region_copy.zones:
                            id_rpc_final = region_copy.id + '_' + zone_copy
                            if id_rpc_final in self.rpc_times[id_rpc]:
                                continue
                            logging.info(f"<PreSchedulerManager>: Testing with zone {zone_copy} of "
                                         f"region {region_copy.region} of provider {region_copy.provider}")
                            if not vm_initial.failed_to_created:
                                vm_initial.deploy(zone=zone, needs_volume=False, key_name=key_file_initial,
                                                  type_task='server')
                                if not vm_initial.failed_to_created:
                                    vm_initial.update_ip(zone=zone)
                            vm_final.zone = zone_copy
                            vm_final.deploy(zone=zone_copy, needs_volume=False, key_name=key_file, type_task='server')
                            if not vm_final.failed_to_created:
                                # update instance IP
                                vm_final.update_ip(zone=zone_copy)
                                self.rpc_times[id_rpc][id_rpc_final] = self.__exec_rpc_vms(vm_initial, vm_final,
                                                                                           region.key_file,
                                                                                           region_copy.key_file)
                                status = vm_final.terminate(wait=True, zone=zone_copy)
                                if status:
                                    vm_final.instance_id = None
                                    vm_final.failed_to_created = False
                                    vm_final.ssh = None
                            else:
                                vm_final.instance_id = None
                                vm_final.failed_to_created = False
                                vm_final.ssh = None
                    loc_copy[region_id].zones.remove(zone)
                    if not vm_initial.failed_to_created:
                        status = vm_initial.terminate(wait=True, zone=zone)
                        if status:
                            vm_initial.instance_id = None
                            vm_initial.failed_to_created = False
                            vm_initial.ssh = None
                    else:
                        vm_initial.instance_id = None
                        vm_initial.failed_to_created = False
                        vm_initial.ssh = None
                loc_copy.pop(region_id, None)
        except Exception as e:
            logging.error(f'<PreSchedulerManager>: Error inside calculate_rpc_times')
            logging.error(e)

    def __exec_rpc_vms(self, vm_server, vm_client, key_server, key_client):

        times = {}

        server_configured = False

        if vm_server.ssh is not None:
            server_configured = True
        else:
            # Start a new SSH Client
            if vm_server.instance_type.provider == CloudManager.EC2:
                vm_server.ssh = SSHClient(vm_server.instance_public_ip, self.loader.ec2_conf.key_path,
                                          key_server, self.loader.ec2_conf.vm_user)
            elif vm_server.instance_type.provider == CloudManager.GCLOUD:
                vm_server.ssh = SSHClient(vm_server.instance_public_ip, self.loader.gcp_conf.key_path,
                                          key_server, self.loader.gcp_conf.vm_user)

        # Start a new SSH Client
        if vm_client.instance_type.provider == CloudManager.EC2:
            vm_client.ssh = SSHClient(vm_client.instance_public_ip, self.loader.ec2_conf.key_path,
                                      key_client, self.loader.ec2_conf.vm_user)
        elif vm_client.instance_type.provider == CloudManager.GCLOUD:
            vm_client.ssh = SSHClient(vm_client.instance_public_ip, self.loader.gcp_conf.key_path,
                                      key_client, self.loader.gcp_conf.vm_user)

        # try to open the connection
        if vm_server.ssh.open_connection() and vm_client.ssh.open_connection():

            rpc_app_item = self.loader.pre_sched_conf.rpc_file
            rpc_client_item = self.loader.pre_sched_conf.client_file
            rpc_server_item = self.loader.pre_sched_conf.server_file

            try:
                logging.info(f"<VirtualMachine {vm_server.instance_id}>: - Sending RPC Files test")

                # Send files
                if vm_server.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    if not server_configured:
                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.ec2_conf.home_path,
                                               item=rpc_app_item)

                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.ec2_conf.home_path,
                                               item=rpc_client_item)

                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.ec2_conf.home_path,
                                               item=rpc_server_item)

                        cmd1 = f'unzip {rpc_app_item} -d .'

                        logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id, cmd1))

                        stdout, stderr, code_return = vm_server.ssh.execute_command(cmd1, output=True)
                        print(stdout)
                    else:
                        cmd_json = f'rm {self.loader.pre_sched_conf.results_temp_file}'

                        logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id,
                                                                                         cmd_json))

                        stdout, stderr, code_return = vm_server.ssh.execute_command(cmd_json, output=True)
                        print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                     self.loader.pre_sched_conf.server_file),
                                                        self.loader.application_conf.fl_port
                                                        )

                elif vm_server.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    if not server_configured:
                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.gcp_conf.home_path,
                                               item=rpc_app_item)

                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.gcp_conf.home_path,
                                               item=rpc_client_item)

                        vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                               target=self.loader.gcp_conf.home_path,
                                               item=rpc_server_item)

                        cmd1 = f'unzip {rpc_app_item} -d .'

                        logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id, cmd1))

                        stdout, stderr, code_return = vm_server.ssh.execute_command(cmd1, output=True)
                        print(stdout)
                    else:
                        cmd_json = f'rm {self.loader.pre_sched_conf.results_temp_file}'

                        logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id,
                                                                                         cmd_json))

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                     self.loader.pre_sched_conf.server_file),
                                                        self.loader.application_conf.fl_port
                                                        )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_server.instance_id))

                stdout, stderr, code_return = vm_server.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                time.sleep(30)

                server_ip = f"{vm_server.instance_public_ip}:{self.loader.application_conf.fl_port}"

                logging.info(f"<VirtualMachine {vm_client.instance_id}>: - Sending RPC Files test")

                # Send files
                if vm_client.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_app_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_client_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_server_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_client.instance_id, cmd1))

                    stdout, stderr, code_return = vm_client.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          self.loader.pre_sched_conf.results_temp_file
                                                          )

                elif vm_client.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_app_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_client_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_server_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_client.instance_id, cmd1))

                    stdout, stderr, code_return = vm_client.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          self.loader.pre_sched_conf.results_temp_file
                                                          )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_client.instance_id))

                stdout, stderr, code_return = vm_client.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                while not has_command_finished(vm_client):
                    continue

                try:
                    if vm_client.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                        vm_client.ssh.get_file(source=vm_client.loader.ec2_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=self.loader.pre_sched_conf.results_temp_file)
                    elif vm_client.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                        vm_client.ssh.get_file(source=vm_client.loader.gcp_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=self.loader.pre_sched_conf.results_temp_file)

                    with open(os.path.join(self.loader.pre_sched_conf.path,
                                           self.loader.pre_sched_conf.results_temp_file)) as f:
                        data = f.read()
                    times['server-client'] = json.loads(data)
                except Exception as e:
                    logging.error(e)
                    times['server-client'] = {}

                # Client-server

                if vm_client.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                     self.loader.pre_sched_conf.server_file),
                                                        self.loader.application_conf.fl_port
                                                        )

                elif vm_client.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                     self.loader.pre_sched_conf.server_file),
                                                        self.loader.application_conf.fl_port
                                                        )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_client.instance_id))

                stdout, stderr, code_return = vm_client.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                time.sleep(30)

                server_ip = f"{vm_client.instance_public_ip}:{self.loader.application_conf.fl_port}"

                if vm_server.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          self.loader.pre_sched_conf.results_temp_file
                                                          )

                elif vm_server.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          self.loader.pre_sched_conf.results_temp_file
                                                          )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_server.instance_id))

                stdout, stderr, code_return = vm_server.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                while not has_command_finished(vm_server):
                    continue

                try:
                    if vm_server.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                        vm_server.ssh.get_file(source=vm_server.loader.ec2_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=self.loader.pre_sched_conf.results_temp_file)
                    elif vm_server.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                        vm_server.ssh.get_file(source=vm_server.loader.gcp_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=self.loader.pre_sched_conf.results_temp_file)

                    with open(os.path.join(self.loader.pre_sched_conf.path,
                                           self.loader.pre_sched_conf.results_temp_file)) as f:
                        data = f.read()
                    times['client-server'] = json.loads(data)
                except Exception as e:
                    logging.error(e)
                    times['client-server'] = {}

                vm_server.ssh.close_connection()
                vm_client.ssh.close_connection()

                return times
            except Exception as e:

                logging.error(e)
                return times

        else:

            logging.error("<PreScheduler> VirtualMachine {} or {}:: "
                          "SSH CONNECTION ERROR".format(vm_server.instance_id, vm_client.instance_id))
            raise Exception("<PreScheduler> VirtualMachine {} or {}:: "
                            "SSH Exception ERROR".format(vm_server.instance_id, vm_client.instance_id))

    def calculate_concurrent_rpc_times(self):
        for num_clients in range(2, self.loader.num_clients_pre_sched+1):
            logging.info("<PreSchedulerManager>: Calculating concurrent RPC times with {} clients".format(num_clients))
            if str(num_clients) not in self.rpc_times_concurrent:
                self.rpc_times_concurrent[str(num_clients)] = {}
            for region_id, region in self.loader.loc.items():
                if region.provider in (CloudManager.EC2, CloudManager.AWS):
                    vm_initial = VirtualMachine(instance_type=instance_aws_rpc_concurrent_server, market='on-demand',
                                                loader=self.loader)
                elif region.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                    vm_initial = VirtualMachine(instance_type=instance_gcp_rpc, market='on-demand', loader=self.loader)
                else:
                    logging.error(f"<PreSchedulingManager>: {region.provider} does not have support ({region_id})")
                    return
                key_file_initial = ''
                if region.key_file != '':
                    key_file_initial = region.key_file.split('.')[0]
                for zone in region.zones:
                    id_rpc = region_id + '_' + zone
                    logging.info(f"<PreSchedulerManager>: Initialing zone {zone} of region {region.region}"
                                 f" or provider {region.provider}")
                    vm_initial.instance_type.image_id = region.server_image_id
                    vm_initial.region = region
                    vm_initial.zone = zone
                    if id_rpc not in self.rpc_times_concurrent[str(num_clients)]:
                        self.rpc_times_concurrent[str(num_clients)][id_rpc] = {}
                    for region_copy in self.loader.loc.values():
                        vms_clients = {}
                        for i in range(num_clients):
                            if region_copy.provider in (CloudManager.EC2, CloudManager.AWS):
                                vms_clients[i] = VirtualMachine(instance_type=instance_aws_rpc, market='on-demand',
                                                                loader=self.loader)
                            elif region_copy.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                                vms_clients[i] = VirtualMachine(instance_type=instance_gcp_rpc, market='on-demand',
                                                                loader=self.loader)
                            else:
                                logging.error(
                                    f"<PreSchedulingManager>: {region_copy.provider} does not have support "
                                    f"({region_copy.id})")
                                return
                            vms_clients[i].instance_type.image_id = region_copy.server_image_id
                            vms_clients[i].region = region_copy
                        for zone_copy in region_copy.zones:
                            id_rpc_final = region_copy.id + '_' + zone_copy
                            if id_rpc_final in self.rpc_times_concurrent[str(num_clients)][id_rpc]:
                                continue
                            for i in range(num_clients):
                                vms_clients[i].zone = zone_copy
                            logging.info(f"<PreSchedulerManager>: Testing with zone {zone_copy} of "
                                         f"region {region_copy.region} of provider {region_copy.provider}")
                            if not vm_initial.failed_to_created:
                                vm_initial.deploy(zone=zone, needs_volume=False, key_name=key_file_initial,
                                                  type_task='server')
                                if not vm_initial.failed_to_created:
                                    vm_initial.update_ip(zone=zone)
                            self.rpc_times_concurrent[str(num_clients)][id_rpc][id_rpc_final] = \
                                self.__exec_concurrent_rpc_vms(vm_initial, vms_clients,
                                                               region, region_copy, num_clients)
                    if not vm_initial.failed_to_created:
                        status = vm_initial.terminate(wait=False, zone=zone)
                        if status:
                            vm_initial.instance_id = None
                            vm_initial.failed_to_created = False
                            vm_initial.ssh = None
                    else:
                        vm_initial.instance_id = None
                        vm_initial.failed_to_created = False
                        vm_initial.ssh = None

    def __exec_concurrent_rpc_vms(self, vm_server, vms_clients, region_server, region_clients, num_clients):
        times = {}

        # Start a new SSH Client
        if vm_server.instance_type.provider == CloudManager.EC2:
            vm_server.ssh = SSHClient(vm_server.instance_public_ip, self.loader.ec2_conf.key_path,
                                      region_server.key_file, self.loader.ec2_conf.vm_user)
        elif vm_server.instance_type.provider == CloudManager.GCLOUD:
            vm_server.ssh = SSHClient(vm_server.instance_public_ip, self.loader.gcp_conf.key_path,
                                      region_server.key_file, self.loader.gcp_conf.vm_user)

        if vm_server.ssh.open_connection():

            rpc_app_item = self.loader.pre_sched_conf.rpc_file
            rpc_server_item = self.loader.pre_sched_conf.server_file

            try:
                logging.info(f"<VirtualMachine {vm_server.instance_id}>: - Sending RPC Files test")

                # Send files
                if vm_server.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_app_item)

                    vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_server_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info(
                        "<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id, cmd1))

                    stdout, stderr, code_return = vm_server.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} " \
                                 "--num_clients {}".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                        self.loader.pre_sched_conf.server_file),
                                                           self.loader.application_conf.fl_port,
                                                           num_clients
                                                           )

                elif vm_server.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_app_item)

                    vm_server.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_server_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info(
                        "<PreScheduling - VirtualMachine {}>: - {} ".format(vm_server.instance_id, cmd1))

                    stdout, stderr, code_return = vm_server.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--fl_port {} " \
                                 "--num_clients {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                         self.loader.pre_sched_conf.server_file),
                                                            self.loader.application_conf.fl_port,
                                                            num_clients
                                                            )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_server.instance_id))

                stdout, stderr, code_return = vm_server.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                server_ip = f"{vm_server.instance_public_ip}:{self.loader.application_conf.fl_port}"

                threads: List[threading.Thread] = []

                times_threads = [None] * num_clients

                for i in range(num_clients):

                    x = threading.Thread(target=self.__exec_concurrent_clients, args=(vms_clients[i],
                                                                                      region_clients,
                                                                                      server_ip,
                                                                                      times_threads,
                                                                                      i))
                    threads.append(x)
                    x.start()

                for i in range(num_clients):
                    threads[i].join()

                i = 0
                for time_client in times_threads:
                    times[i] = time_client
                    i = i + 1

                # print('times_threads', times)

            except Exception as e:
                logging.error(e)

            status = vm_server.terminate(wait=False, zone=vm_server.zone)
            if status:
                vm_server.instance_id = None
                vm_server.failed_to_created = False
                vm_server.ssh = None

            return times

        else:

            logging.error("<PreScheduler> VirtualMachine {}:: "
                          "SSH CONNECTION ERROR".format(vm_server.instance_id))
            raise Exception("<PreScheduler> VirtualMachine {}:: "
                            "SSH Exception ERROR".format(vm_server.instance_id))

    def __exec_concurrent_clients(self, vm_client, region_client, server_ip, results, num_client):
        results[num_client] = {}
        key_file = ''
        if region_client.key_file != '':
            key_file = region_client.key_file.split('.')[0]

        vm_client.deploy(zone=vm_client.zone, needs_volume=False, key_name=key_file, type_task='server')
        if not vm_client.failed_to_created:
            # update instance IP
            vm_client.update_ip(zone=vm_client.zone)

            # Start a new SSH Client
            if vm_client.instance_type.provider == CloudManager.EC2:
                vm_client.ssh = SSHClient(vm_client.instance_public_ip, self.loader.ec2_conf.key_path,
                                          region_client.key_file, self.loader.ec2_conf.vm_user)
            elif vm_client.instance_type.provider == CloudManager.GCLOUD:
                vm_client.ssh = SSHClient(vm_client.instance_public_ip, self.loader.gcp_conf.key_path,
                                          region_client.key_file, self.loader.gcp_conf.vm_user)

            # try to open the connection
            if vm_client.ssh.open_connection():
                rpc_app_item = self.loader.pre_sched_conf.rpc_file
                rpc_client_item = self.loader.pre_sched_conf.client_file

                temp_file = str(num_client) + "_" + self.loader.pre_sched_conf.results_temp_file

                logging.info(f"<VirtualMachine {vm_client.instance_id}>: - Sending RPC Files test")

                # Send files
                if vm_client.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_app_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.ec2_conf.home_path,
                                           item=rpc_client_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_client.instance_id, cmd1))

                    stdout, stderr, code_return = vm_client.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          temp_file
                                                          )

                elif vm_client.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_app_item)

                    vm_client.ssh.put_file(source=self.loader.pre_sched_conf.path,
                                           target=self.loader.gcp_conf.home_path,
                                           item=rpc_client_item)

                    cmd1 = f'unzip {rpc_app_item} -d .'

                    logging.info("<PreScheduling - VirtualMachine {}>: - {} ".format(vm_client.instance_id, cmd1))

                    stdout, stderr, code_return = vm_client.ssh.execute_command(cmd1, output=True)
                    print(stdout)

                    cmd_daemon = "python3.7 {} " \
                                 "--server_address {} " \
                                 "--length_parameters {} " \
                                 "--save_file {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                       self.loader.pre_sched_conf.client_file),
                                                          server_ip,
                                                          self.loader.pre_sched_conf.length_msg,
                                                          temp_file
                                                          )
                else:
                    cmd_daemon = ""

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{} "'.format(cmd_daemon)
                logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {} ".format(cmd_screen,
                                                                                             vm_client.instance_id))

                stdout, stderr, code_return = vm_client.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                while not has_command_finished(vm_client):
                    continue

                try:
                    if vm_client.instance_type.provider in (CloudManager.EC2, CloudManager.AWS):
                        vm_client.ssh.get_file(source=vm_client.loader.ec2_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=temp_file)
                    elif vm_client.instance_type.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                        vm_client.ssh.get_file(source=vm_client.loader.gcp_conf.home_path,
                                               target=self.loader.pre_sched_conf.path,
                                               item=temp_file)

                    with open(os.path.join(self.loader.pre_sched_conf.path,
                                           temp_file)) as f:
                        data = f.read()
                    results[num_client] = json.loads(data)
                    # print("thread", num_client, "results", results[num_client])
                except Exception as e:
                    logging.error(e)
                    results[num_client] = {}

                status = vm_client.terminate(wait=False, zone=vm_client.zone)
                if status:
                    vm_client.instance_id = None
                    vm_client.failed_to_created = False
                    vm_client.ssh = None
            else:

                logging.error("<PreScheduler> VirtualMachine {}:: "
                              "SSH CONNECTION ERROR".format(vm_client.instance_id))
                raise Exception("<PreScheduler> VirtualMachine {}:: "
                                "SSH Exception ERROR".format(vm_client.instance_id))
        else:
            vm_client.instance_id = None
            vm_client.failed_to_created = False
            vm_client.ssh = None

    def write_json(self):
        # build a map
        dict_json = {"job_id": self.loader.job.job_id,
                     "job_name": self.loader.job.job_name,
                     "number_datacenters": len(self.rtt_values),
                     "rtt": {},
                     "exec_times": self.exec_times,
                     "rpc": {},
                     "rpc_concurrent": {}
                     }

        # print("dict_json")
        # print(dict_json)

        for datacenter, rtt_value in self.rtt_values.items():
            # print(datacenter)
            # print("rtt_value", rtt_value)
            dict_json['rtt'][datacenter] = rtt_value

        for datacenter, rpc_value in self.rpc_times.items():
            # print(datacenter)
            # print("rpc_value", rpc_value)
            dict_json['rpc'][datacenter] = rpc_value

        for num_clients, rpc_value in self.rpc_times_concurrent.items():
            # print(datacenter)
            # print("rpc_value", rpc_value)
            dict_json['rpc_concurrent'][num_clients] = rpc_value

        file_output = self.loader.pre_file

        # print(dict_json)
        logging.info(f"<PreSchedulerManager> Writing {file_output} file")

        # print("dict_json", dict_json)

        with open(file_output, "w") as fp:
            json.dump(dict_json, fp, sort_keys=True, indent=4, default=str)

    def __read_json(self):
        if os.path.exists(self.loader.pre_file):
            logging.info(f"<PreSchedulerManager> File {self.loader.pre_file} already exists. Reading info")
            try:
                with open(self.loader.pre_file) as f:
                    data = f.read()
                json_data = json.loads(data)
                self.exec_times = json_data['exec_times']
                self.rtt_values = json_data['rtt']
                self.rpc_times = json_data['rpc']
                self.rpc_times_concurrent = json_data['rpc_concurrent']
                if json_data['job_id'] != self.loader.job.job_id or json_data['job_name'] != self.loader.job.job_name:
                    logging.error(f"<PreSchedulerManager> Current file {self.loader.pre_file} is not for this job!")
                    rep = str(input("Do you want to stop execution? [N] for no; otherwise yes"))
                    return rep.upper() != 'N'

                # print("self.exec_times:")
                # print(json.dumps(self.exec_times, indent=4, sort_keys=True))
                #
                # print("self.rtt_values:")
                # print(json.dumps(self.rtt_values, indent=4, sort_keys=True))
                # 
                # print("self.rpc_times_concurrent:")
                # print(json.dumps(self.rpc_times_concurrent, indent=4, sort_keys=True))

            except Exception as e:
                logging.error(e)

        return False
