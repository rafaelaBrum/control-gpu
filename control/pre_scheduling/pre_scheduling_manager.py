import json
import logging
import os
import time
from typing import Dict, List
from copy import deepcopy

from control.domain.instance_type import InstanceType
from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine
from control.util.loader import Loader
from control.util.ssh_client import SSHClient

instance_aws = InstanceType(
    provider=CloudManager.EC2,
    instance_type='t2.micro',
    image_id='ami-0cdc662c42e7c28ed',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    ebs_device_name='/dev/xvdf'
)
instance_gcp = InstanceType(
    provider=CloudManager.GCLOUD,
    instance_type='e2-micro',
    image_id='disk-ubuntu-flower-server',
    restrictions={'on-demand': 1,
                  'preemptible': 1},
    prices={'on-demand': 0.001,
            'preemptible': 0.000031},
    memory=8,
    vcpu=2,
    ebs_device_name='/dev/sdb'
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
        self.exec_times: Dict[str, List[float]] = {}

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
                vm_initial.zone = zone
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
                    key_file = ''
                    if region_copy.key_file != '':
                        key_file = region_copy.key_file.split('.')[0]
                    for zone_copy in region_copy.zones:
                        logging.info(f"<PreSchedulerManager>: Testing with zone {zone_copy} of "
                                     f"region {region_copy.region} of provider {region_copy.provider}")
                        id_rtt_final = region_copy.id + '_' + zone_copy
                        if region.id == region_id and zone_copy == zone:
                            continue
                        if not vm_initial.failed_to_created:
                            vm_initial.deploy(zone=zone, needs_volume=False, key_name=key_file_initial)
                            if not vm_initial.failed_to_created:
                                vm_initial.update_ip(zone=zone)
                        vm_final.zone = zone_copy
                        vm_final.deploy(zone=zone_copy, needs_volume=False, key_name=key_file)
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

                vm_final.ssh.put_file(source=self.loader.pre_sched_conf.rtt_path,
                                      target=self.loader.ec2_conf.home_path,
                                      item=item)

                vm_final.ssh.put_file(source=key_path,
                                      target=self.loader.ec2_conf.home_path,
                                      item=item_key)

                cmd_daemon = "python3 {} " \
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

                vm_final.ssh.put_file(source=self.loader.pre_sched_conf.rtt_path,
                                      target=self.loader.gcp_conf.home_path,
                                      item=item)

                vm_final.ssh.put_file(source=key_path,
                                      target=self.loader.gcp_conf.home_path,
                                      item=item_key)

                cmd_daemon = "python3 {} " \
                             "--ip {} " \
                             "--path {} " \
                             "--file {} " \
                             "--user {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                              self.loader.pre_sched_conf.rtt_file),
                                                 vm_initial.instance_public_ip,
                                                 self.loader.gcp_conf.home_path,
                                                 item_key,
                                                 vm_user)
            else:
                cmd_daemon = ""

            cmd_screen = 'screen -L -Logfile $HOME/screen_log -S test -dm bash -c "{}"'.format(cmd_daemon)
            logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {}".format(cmd_screen,
                                                                                        vm_final.instance_id))

            stdout, stderr, code_return = vm_final.ssh.execute_command(cmd_screen, output=True)
            print(stdout)

            while not has_command_finished(vm_final):
                continue

            cmd = "cat $HOME/screen_log"
            logging.info("<PreScheduler>: - Executing '{}' on VirtualMachine {}".format(cmd,
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

    @staticmethod
    def write_json(rtt_values: Dict[str, Dict[str, float]], exec_times: Dict[str, List[float]], job_id, job_name,
                   file_output):
        # build a map
        dict_json = {"job_id": job_id,
                     "job_name": job_name,
                     "number_datacenters": len(rtt_values),
                     "rtt": {},
                     "exec_times": exec_times
                     }

        for datacenter, rtt_value in rtt_values.items():
            # print(datacenter)
            # print("rtt_value", rtt_value)
            dict_json['rtt'][datacenter] = rtt_value

        # print(dict_json)
        logging.info(f"<PreSchedulerManager> Writing {file_output} file")

        with open(file_output, "w") as fp:
            json.dump(dict_json, fp, sort_keys=True, indent=4, default=str)
