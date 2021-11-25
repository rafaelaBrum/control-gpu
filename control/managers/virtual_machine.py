from control.domain.instance_type import InstanceType

from control.managers.cloud_manager import CloudManager
from control.managers.ec2_manager import EC2Manager
from control.managers.gcp_manager import GCPManager

from control.util.ssh_client import SSHClient
from control.util.loader import Loader

from datetime import datetime, timedelta

import uuid
import logging

import os

# Class to control the virtual machine in the cloud
# Parameters: Instance_type (InstanceType), market (str), primary (boolean)

# If primary flag is True, it indicates that the VM is a primary resource, i.e,
# the VM was launched to executed a primary task.
# Otherwise, if primary flag is False, the VM is a backup resource launched due to spot interruption/hibernation

class VirtualMachine:
    CPU_BURST = -1

    def __init__(self, instance_type: InstanceType, market, loader: Loader, volume_id=None, disk_name=''):

        self.loader = loader

        self.instance_type = instance_type
        self.market = market
        self.volume_id = volume_id
        self.disk_name = disk_name

        self.create_file_system = False

        # Start cloud manager
        if instance_type.provider == CloudManager.EC2:
            self.manager = EC2Manager()
        elif instance_type.provider == CloudManager.GCLOUD:
            self.manager = GCPManager()

        self.instance_id = None
        self.instance_public_ip = None
        self.instance_private_ip = None
        self.current_state = CloudManager.PENDING
        self.marked_to_interrupt = False

        self.ready = False

        # Simulation
        self.in_simulation = False

        self.ssh: SSHClient = None  # SSH Client

        # Time tracker vars
        self.__start_time = datetime.now()
        self.__end_time = datetime.now()

        self.__start_time_utc = datetime.utcnow()

        self.deploy_overhead = timedelta(seconds=0)

        self.start_deploy = datetime.now()
        self.terminate_overhead = timedelta(seconds=0)

        # Hibernation Trackers
        self.hibernation_process_finished = False

        self.__hibernation_duration = timedelta(seconds=0)
        self.__hibernation_start_timestamp = None
        self.__hibernation_end_timestamp = None

        self.failed_to_created = False

        self.root_folder = None

    '''
        Methods to Manager the virtual Machine
    '''

    # Start the virtual machine
    # Return (boolean) True if success otherwise return False
    def deploy(self):

        self.start_deploy = datetime.now()

        # Check if a VM was already created
        if self.instance_id is None:

            logging.info("<VirtualMachine>: Deploying a new {} instance of type {} "
                         .format(self.market, self.instance_type.type))

            try:

                if self.market not in (CloudManager.ON_DEMAND, CloudManager.PREEMPTIBLE):
                    raise Exception("<VirtualMachine>: Invalid Market - {}:".format(self.market))
                elif self.market == CloudManager.ON_DEMAND and self.instance_type.provider == CloudManager.EC2:
                    self.instance_id = self.manager.create_on_demand_instance(instance_type=self.instance_type.type,
                                                                              image_id=self.instance_type.image_id)
                elif self.market == CloudManager.ON_DEMAND and self.instance_type.provider == CloudManager.GCLOUD:
                    self.instance_id = self.manager.create_on_demand_instance(instance_type=self.instance_type.type,
                                                                              image_id=self.instance_type.image_id,
                                                                              vm_name=self.instance_type.vm_name)
                elif self.market == CloudManager.PREEMPTIBLE and self.instance_type.provider == CloudManager.EC2:
                    self.instance_id = \
                        self.manager.create_preemptible_instance(instance_type=self.instance_type.type,
                                                                 image_id=self.instance_type.image_id,
                                                                 max_price=self.instance_type.price_preemptible + 0.1)
                else:
                    raise Exception(f"<VirtualMachine>: We do not support {self.market} instances on "
                                    f"{self.instance_type.provider} cloud provider yet")

            except Exception as e:
                logging.error("<VirtualMachine>: "
                              "Error to create  {} instance of type {} ".format(self.market,
                                                                                self.instance_type.type))
                self.instance_id = None
                logging.error(e)

            # check if instance was create with success
            if self.instance_id is not None:
                logging.info("<VirtualMachine {}>: Market: {} Type: {} Provider: {}"
                             " Create with Success".format(self.instance_id,
                                                           self.market,
                                                           self.instance_type.type,
                                                           self.instance_type.provider))

                # update start_times
                self.__start_time = datetime.now()
                self.__start_time_utc = datetime.utcnow()

                self.instance_public_ip = self.manager.get_public_instance_ip(self.instance_id)
                self.instance_private_ip = self.manager.get_private_instance_ip(self.instance_id)

                if self.loader.file_system_conf.type == CloudManager.EBS:
                    # if there is not a volume create a new volume
                    if self.volume_id is None:
                        self.volume_id = self.manager.create_volume(
                            size=self.loader.file_system_conf.size,
                            volume_name=self.disk_name
                        )
                        self.create_file_system = True

                        if self.volume_id is None:
                            raise Exception(
                                "<VirtualMachine {}>: :Error to create new volume".format(self.instance_id))

                    logging.info(
                        "<VirtualMachine {}>: Attaching Volume {}".format(self.instance_id, self.volume_id))
                    if self.instance_type.provider == CloudManager.EC2:
                        # attached new volume
                        # waiting until volume available
                        self.manager.wait_volume(volume_id=self.volume_id)
                        self.manager.attach_volume(
                            instance_id=self.instance_id,
                            volume_id=self.volume_id
                        )
                    elif self.instance_type.provider == CloudManager.GCLOUD:
                        # attached new volume
                        # waiting until volume available
                        self.manager.wait_volume(volume_name=self.disk_name)
                        self.manager.attach_volume(
                            instance_id=self.instance_id,
                            volume_id=self.volume_id,
                            volume_name=self.disk_name
                        )

                return True

            else:

                self.instance_id = 'f-{}'.format(str(uuid.uuid4())[:8])
                self.failed_to_created = True

                return False
        else:
            self.instance_public_ip = self.manager.get_public_instance_ip(self.instance_id)
            self.instance_private_ip = self.manager.get_private_instance_ip(self.instance_id)

        # Instance was already started
        return False

    def __create_ebs(self, path):

        internal_device_name = self.instance_type.ebs_device_name

        logging.info("<VirtualMachine {}>: - Mounting EBS/GCLOUD DISK".format(self.instance_id))

        if self.create_file_system:
            cmd1 = 'sudo mkfs.ext4 {}'.format(internal_device_name)
            logging.info("<VirtualMachine {}>: - {} ".format(self.instance_id, cmd1))
            self.ssh.execute_command(cmd1, output=True)

        # Mount Directory
        cmd2 = 'sudo mount {} {}'.format(internal_device_name, path)
        logging.info("<VirtualMachine {}>: - {} ".format(self.instance_id, cmd2))

        self.ssh.execute_command(cmd2, output=True)  # mount directory

        if self.create_file_system:
            cmd3 = 'sudo chown {0}:{0} -R {1}'.format(self.manager.vm_config.vm_user,
                                                      self.loader.file_system_conf.path)
            cmd4 = 'chmod 775 -R {}'.format(self.loader.file_system_conf.path)

            logging.info("<VirtualMachine {}>: - {} ".format(self.instance_id, cmd3))
            self.ssh.execute_command(cmd3, output=True)

            logging.info("<VirtualMachine {}>: - {} ".format(self.instance_id, cmd4))
            self.ssh.execute_command(cmd4, output=True)

    def __create_s3(self, path):

        logging.info("<VirtualMachine {}>: - Mounting S3FS".format(self.instance_id))

        # prepare S3FS
        cmd1 = 'echo {}:{} > $HOME/.passwd-s3fs'.format(self.manager.credentials.access_key,
                                                        self.manager.credentials.secret_key)

        cmd2 = 'sudo chmod 600 $HOME/.passwd-s3fs'

        # Mount the bucket
        cmd3 = 'sudo s3fs {} ' \
               '-o use_cache=/tmp -o allow_other -o uid={} -o gid={} ' \
               '-o mp_umask=002 -o multireq_max=5 {}'.format(self.manager.s3_conf.bucket_name,
                                                             self.manager.s3_conf.vm_uid,
                                                             self.manager.s3_conf.vm_gid,
                                                             path)

        logging.info("<VirtualMachine {}>: - Creating .passwd-s3fs".format(self.instance_id))
        self.ssh.execute_command(cmd1, output=True)

        logging.info("<VirtualMachine {}>: - {}".format(self.instance_id, cmd2))
        self.ssh.execute_command(cmd2, output=True)

        logging.info("<VirtualMachine {}>: - {}".format(self.instance_id, cmd3))
        self.ssh.execute_command(cmd3, output=True)

    def __create_google_storage(self, path):

        logging.info("<VirtualMachine {}>: - Mounting GCSFUSE".format(self.instance_id))

        # Mount the bucket
        cmd = 'gcsfuse --implicit-dirs {} {}'.format(self.manager.storage_config.bucket_name,
                                                     path)

        logging.info("<VirtualMachine {}>: - {}".format(self.instance_id, cmd))
        self.ssh.execute_command(cmd, output=True)

    def prepare_vm(self):

        if not self.failed_to_created:

            # update instance IP
            self.update_ip()
            # Start a new SSH Client
            if self.instance_type.provider == CloudManager.EC2:
                self.ssh = SSHClient(self.instance_public_ip, self.loader.ec2_conf.key_path,
                                     self.loader.ec2_conf.key_file, self.loader.ec2_conf.vm_user)
            elif self.instance_type.provider == CloudManager.GCLOUD:
                self.ssh = SSHClient(self.instance_public_ip, self.loader.gcp_conf.key_path,
                                     self.loader.gcp_conf.key_file, self.loader.gcp_conf.vm_user)

            # try to open the connection
            if self.ssh.open_connection():

                logging.info("<VirtualMachine {}>: - Creating directory {}".format(self.instance_id,
                                                                                   self.loader.file_system_conf.path))
                # create directory
                #TODO: add storage options to GCP
                self.ssh.execute_command('mkdir -p {}'.format(self.loader.file_system_conf.path), output=True)

                if self.loader.file_system_conf.type == CloudManager.EBS:
                    self.__create_ebs(self.loader.file_system_conf.path)
                elif self.loader.file_system_conf.type == CloudManager.S3:
                    if self.instance_type.provider == CloudManager.EC2:
                        self.__create_s3(self.loader.file_system_conf.path)
                    elif self.instance_type.provider == CloudManager.GCLOUD:
                        self.__create_google_storage(self.loader.file_system_conf.path)
                else:
                    logging.error("<VirtualMachine {}>: - Storage type error".format(self.instance_id))

                    raise Exception(
                        "VM {} Storage {} not supported".format(self.instance_id, self.loader.file_system_conf.type))
                # elif self.instance_type.provider == CloudManager.GCLOUD:
                #     self.ssh.execute_command('mkdir -p {}'.format(self.loader.file_system_conf.path), output=True)
                #
                #     # if self.loader.file_system_conf.type == CloudManager.EBS:
                #     #     self.__create_gcp_disk(self.loader.file_system_conf.path)
                #     # elif self.loader.file_system_conf.type == CloudManager.S3:
                #     #     self.__create_cloud_storage(self.loader.file_system_conf.path)
                #     # else:
                #     #     logging.error("<VirtualMachine {}>: - Storage type error".format(self.instance_id))
                #     #
                #     #     raise Exception(
                #     #         "VM {} Storage {} not supported".format(self.instance_id, self.loader.file_system_conf.type))

                # keep ssh live
                # self.ssh.execute_command("$HOME/.ssh/config")

                cmd = 'mkdir {}_{}/'.format(self.loader.cudalign_task.task_id, self.loader.execution_id)

                logging.info("<VirtualMachine {}>: - {}".format(self.instance_id, cmd))

                stdout, stderr, code_return = self.ssh.execute_command(cmd, output=True)
                print(stdout)

                # Send daemon file
                if self.instance_type.provider == CloudManager.EC2:
                    self.ssh.put_file(source=self.loader.application_conf.daemon_path,
                                      target=self.loader.ec2_conf.home_path,
                                      item=self.loader.application_conf.daemon_file)

                    # create execution folder
                    self.root_folder = os.path.join(self.loader.file_system_conf.path,
                                                    '{}_{}'.format(self.loader.cudalign_task.task_id,
                                                                   self.loader.execution_id))

                    cmd_daemon = "python3 {} " \
                                 "--vm_user {} " \
                                 "--root_path {} " \
                                 "--task_id {} " \
                                 "--execution_id {}  " \
                                 "--instance_id {} ".format(os.path.join(self.loader.ec2_conf.home_path,
                                                                         self.loader.application_conf.daemon_file),
                                                            self.loader.ec2_conf.vm_user,
                                                            self.loader.file_system_conf.path,
                                                            self.loader.cudalign_task.task_id,
                                                            self.loader.execution_id,
                                                            self.instance_id)

                elif self.instance_type.provider == CloudManager.GCLOUD:
                    self.ssh.put_file(source=self.loader.application_conf.daemon_path,
                                      target=self.loader.gcp_conf.home_path,
                                      item=self.loader.application_conf.daemon_file)

                    # create execution folder
                    self.root_folder = os.path.join(self.loader.gcp_conf.home_path,
                                                    '{}_{}'.format(self.loader.cudalign_task.task_id,
                                                                   self.loader.execution_id))

                    cmd_daemon = "python3 {} " \
                                 "--vm_user {} " \
                                 "--root_path {} " \
                                 "--task_id {} " \
                                 "--execution_id {}  " \
                                 "--instance_id {} ".format(os.path.join(self.loader.gcp_conf.home_path,
                                                                         self.loader.application_conf.daemon_file),
                                                            self.loader.gcp_conf.vm_user,
                                                            self.loader.file_system_conf.path,
                                                            self.loader.cudalign_task.task_id,
                                                            self.loader.execution_id,
                                                            self.instance_id)

                self.ssh.execute_command('mkdir -p {}'.format(self.root_folder), output=True)

                # Start Daemon
                logging.info("<VirtualMachine {}>: - Starting Daemon".format(self.instance_id))

                cmd_screen = 'screen -L -Logfile $HOME/screen_log -dm bash -c "{}"'.format(cmd_daemon)
                # cmd_screen = '{}'.format(cmd_daemon)

                logging.info("<VirtualMachine {}>: - {}".format(self.instance_id, cmd_screen))

                stdout, stderr, code_return = self.ssh.execute_command(cmd_screen, output=True)
                print(stdout)

                self.deploy_overhead = datetime.now() - self.start_deploy

            else:

                logging.error("<VirtualMachine {}>:: SSH CONNECTION ERROR".format(self.instance_id))
                raise Exception("<VirtualMachine {}>:: SSH Exception ERROR".format(self.instance_id))

    # Terminate the virtual machine
    # and delete volume (if delete_volume = True)
    # Return True if success otherwise return False
    def terminate(self, delete_volume=True):

        logging.info("<VirtualMachine>: Terminating instance {} ".format(self.instance_id))

        terminate_start = datetime.now()

        status = False

        if self.state not in (CloudManager.TERMINATED, None):
            self.__end_time = datetime.now()
            status = self.manager.terminate_instance(self.instance_id)

            if delete_volume and self.volume_id is not None:
                if self.instance_type.provider == CloudManager.EC2:
                    self.manager.delete_volume(self.volume_id)
                elif self.instance_type.provider == CloudManager.GCLOUD:
                    self.manager.delete_volume(self.volume_id, volume_name=self.disk_name)

        self.terminate_overhead = datetime.now() - terminate_start

        return status

    def update_ip(self):
        self.instance_public_ip = self.manager.get_public_instance_ip(self.instance_id)
        self.instance_private_ip = self.manager.get_private_instance_ip(self.instance_id)

    # return the a IP's list of all running instance on the cloud provider
    def get_instances_ip(self):

        if self.instance_type.provider == CloudManager.EC2:
            filter_instance = {
                'status': [CloudManager.PENDING, CloudManager.RUNNING],
                'tags': [{'Key': self.loader.ec2_conf.tag_key,
                          'Value': self.loader.ec2_conf.tag_value
                          }]
            }
        elif self.instance_type.provider == CloudManager.GCLOUD:
            filter_instance = f'(status = {CloudManager.RUNNING}) OR (status = {CloudManager.PENDING})'

        instances_id = self.manager.list_instances_id(filter_instance)
        ip_list = []
        for id_instance in instances_id:
            ip_list.append(self.manager.get_public_instance_ip(id_instance))

        return ip_list

    def interrupt(self):
        self.marked_to_interrupt = True

    # Return the current state of the virtual machine
    @property
    def state(self):
        if self.marked_to_interrupt:
            self.current_state = CloudManager.STOPPING
        elif not self.in_simulation:
            if not self.failed_to_created:
                self.current_state = self.manager.get_instance_status(self.instance_id)
            else:
                self.current_state = CloudManager.ERROR

            if self.current_state is None:
                self.current_state = CloudManager.TERMINATED

        return self.current_state

    # Return the machine start time
    # Return None If the machine has not start
    @property
    def start_time(self):
        return self.__start_time

    # Return the machine start time UTC
    # Return None If the machine has not start

    @property
    def start_time_utc(self):
        return self.__start_time_utc

    # Return the uptime if the machine was started
    # Otherwise return None
    @property
    def uptime(self):
        _uptime = timedelta(seconds=0)

        # check if the VM has started
        if self.__start_time is not None:
            # check if VM has terminated
            if self.__end_time is not None:
                _uptime = self.__end_time - self.__start_time
            else:
                _uptime = datetime.now() - self.__start_time

            # remove the hibernation_duration
            _uptime = max(_uptime - self.hibernation_duration, timedelta(seconds=0))

        return _uptime

    # Return the shutdown time if the machine was terminated
    # Otherwise return None
    @property
    def end_time(self):
        return self.__end_time

    @property
    def hibernation_duration(self):
        return self.__hibernation_duration

    @property
    def price(self):
        if self.instance_type.provider == CloudManager.EC2:
            if self.market == CloudManager.PREEMPTIBLE:
                return self.manager.get_preemptible_price(self.instance_type.type, self.loader.ec2_conf.zone)[0][1]
            else:
                return self.manager.get_ondemand_price(self.instance_type.type, self.loader.ec2_conf.region)
        elif self.instance_type.provider == CloudManager.GCLOUD:
            if self.market == CloudManager.PREEMPTIBLE:
                vcpu_price, mem_price = self.manager.get_preemptible_price(self.instance_type.type, self.loader.gcp_conf.region)
            else:
                vcpu_price, mem_price = self.manager.get_ondemand_price(self.instance_type.type, self.loader.gcp_conf.region)
            return self.instance_type.vcpus*vcpu_price + self.instance_type.memory*mem_price

    @property
    def type(self):
        return self.instance_type.type
