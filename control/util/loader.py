#!/usr/bin/env python


# from control.managers.ec2_manager import EC2Manager
from control.managers.cloud_manager import CloudManager
from control.managers.gcp_manager import GCPManager

from control.config.application_config import ApplicationConfig
from control.config.checkpoint_config import CheckPointConfig
from control.config.communication_config import CommunicationConfig
from control.config.database_config import DataBaseConfig
from control.config.debug_config import DebugConfig
from control.config.ec2_config import EC2Config
from control.config.gcp_config import GCPConfig
from control.config.file_system_config import FileSystemConfig
from control.config.input_config import InputConfig
from control.config.logging_config import LoggingConfig
from control.config.notify_config import NotifyConfig
from control.config.simulation_config import SimulationConfig
from control.config.pre_scheduling_config import PreSchedConfig
from control.config.cloudlab_config import CloudLabConfig
from control.config.mapping_config import MappingConfig

from control.domain.instance_type import InstanceType
from control.domain.job import Job
from control.domain.cloud_region import CloudRegion

from control.repository.postgres_repo import PostgresRepo

from datetime import timedelta

from typing import Dict

import logging
import json
import os

from pathlib import Path


class Loader:
    VERSION = '0.0.1'

    env: Dict[str, InstanceType]
    job: Job
    loc: Dict[str, CloudRegion]

    def __init__(self, args):

        # print("Clients provider", args.clients_provider)
        # print("Clients region", args.clients_region)
        # print("Clients VM name", args.clients_vm_name)
        #
        # print("Server provider", args.server_provider)
        # print("Server region", args.server_region)
        # print("Server VM name", args.server_vm_name)

        """Command line args"""
        # input files parameters
        self.input_path = args.input_path
        self.job_file = args.job_file
        self.env_file = args.env_file
        self.loc_file = args.loc_file
        self.pre_file = args.pre_file
        self.input_file = args.input_file
        self.map_file = args.map_file

        # deadline in seconds parameter
        self.deadline_seconds = args.deadline_seconds
        # # ac size in seconds
        # self.ac_size_seconds = args.ac_size_seconds
        # log file name
        self.log_file = args.log_file
        # # simulation parameters
        # self.resume_rate = args.resume_rate
        self.revocation_rate = args.revocation_rate
        # # name of the scheduler
        # self.scheduler_name = args.scheduler_name
        # # notify end of execution by email
        # self.notify = args.notify
        # server provider and region
        self.server_provider = args.server_provider
        self.server_region = args.server_region
        self.server_vm_name = args.server_vm_name
        # clients provider and region
        self.clients_provider = args.clients_provider
        self.clients_region = args.clients_region
        self.clients_vm_name = args.clients_vm_name
        # Client command
        self.client_command = args.command
        # getting scheduler name
        self.scheduler_name = args.scheduler_name
        self.frequency_ckpt = args.frequency_ckpt

        self.emulated = args.emulated

        self.num_clients_pre_scheduling = args.num_clients_pre_scheduling

        self.deadline_timedelta = None

        # instances able to be used
        self.instances_list = []

        # limits Parameters
        # global on-demand count_list
        self.count_list = {}
        self.max_preemptible = None
        self.max_on_demand = None
        # used to attend the cloud limits

        # Load config Classes
        self.application_conf = ApplicationConfig()
        self.checkpoint_conf = CheckPointConfig()
        self.communication_conf = CommunicationConfig()
        self.database_conf = DataBaseConfig()
        self.debug_conf = DebugConfig()
        self.ec2_conf = EC2Config()
        self.gcp_conf = GCPConfig()
        self.file_system_conf = FileSystemConfig()
        self.input_conf = InputConfig()
        self.logging_conf = LoggingConfig()
        self.notify_conf = NotifyConfig()
        self.simulation_conf = SimulationConfig()
        self.pre_scheduling_conf = PreSchedConfig()
        self.cloudlab_conf = CloudLabConfig()
        self.mapping_conf = MappingConfig()

        # local path where the daemon file is
        self.daemon_aws_file = None
        self.daemon_gcp_file = None

        '''
        Parameters of the execution
        The execution_id is defined according with the database last execution_id
        '''
        self.execution_id = None  # ID of the current execution

        '''
        Dictionary with the domain.instance_type that can be used in the execution
        Read from env.json
        '''
        self.env = None

        '''
        Class domain.job contains all tasks that will be executed
        and the information  about the job
        '''
        self.job = None

        '''
        Dictionary with the domain.cloud_region that can be used in the execution
        Read from loc.json
        '''
        self.loc = None

        self.__prepare_logging()
        self.__load_input_parameters()

        self.__load_job()
        self.__load_env()
        self.__load_loc()

        self.__get_execution_id()

        self.__update_prices()
        self.__update_zones()

        self.__update_command(args.strategy, args.num_seed)

    def __prepare_logging(self):
        """
        Set up the log format, level and the file where it will be recorded.
        """
        if self.log_file is None:
            self.log_file = Path(self.logging_conf.path, self.logging_conf.log_file)
        else:
            self.log_file = Path(self.logging_conf.path, self.log_file)

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logging_conf.level)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    def __load_input_parameters(self):
        """
        Merging command line arguments with arguments from setup.cfg
        """

        if self.input_path is None:
            self.input_path = self.input_conf.path

        if self.job_file is None:
            self.job_file = os.path.join(self.input_path, self.input_conf.job_file)
        else:
            self.job_file = os.path.join(self.input_path, self.job_file)

        if self.env_file is None:
            self.env_file = os.path.join(self.input_path, self.input_conf.env_file)
        else:
            self.env_file = os.path.join(self.input_path, self.env_file)

        if self.loc_file is None:
            self.loc_file = os.path.join(self.input_path, self.input_conf.loc_file)
        else:
            self.loc_file = os.path.join(self.input_path, self.loc_file)

        if self.pre_file is None:
            self.pre_file = os.path.join(self.input_path, self.input_conf.pre_file)
        else:
            self.pre_file = os.path.join(self.input_path, self.pre_file)

        if self.input_file is None:
            self.input_file = os.path.join(self.input_path, self.input_conf.input_file)
        else:
            self.input_file = os.path.join(self.input_path, self.input_file)

        if self.map_file is None:
            self.map_file = os.path.join(self.input_path, self.input_conf.map_file)
        else:
            self.map_file = os.path.join(self.input_path, self.map_file)

        if self.scheduler_name is None:
            self.scheduler_name = self.mapping_conf.scheduler_name

        self.daemon_aws_file = os.path.join(self.application_conf.daemon_path, self.application_conf.daemon_aws_file)
        self.daemon_gcp_file = os.path.join(self.application_conf.daemon_path, self.application_conf.daemon_gcp_file)

        if self.deadline_seconds is None:
            self.deadline_seconds = self.input_conf.deadline_seconds

        self.deadline_timedelta = timedelta(seconds=self.deadline_seconds)

        # if self.ac_size_seconds is None:
        #     self.ac_size_seconds = self.input_conf.ac_size_seconds

        if self.revocation_rate is None:
            self.revocation_rate = self.simulation_conf.revocation_rate

        if self.num_clients_pre_scheduling is None:
            self.num_clients_pre_scheduling = self.pre_scheduling_conf.num_clients

        if self.frequency_ckpt is None:
            self.frequency_ckpt = self.checkpoint_conf.frequency_ckpt

    def __load_job(self):
        """
        Read the file job_file and create the class Job
        """

        try:
            with open(self.job_file) as f:
                job_json = json.load(f)
        except Exception as e:
            logging.error("<Loader>: Error file {} ".format(self.job_file))
            raise Exception(e)

        self.job = Job.from_dict(job_json)

    def __load_env(self):
        """
        Read the file env_file and create a dictionary with all available instances
        """

        try:
            with open(self.env_file) as f:
                env_json = json.load(f)
        except Exception as e:
            logging.error("<Loader>: Error file {} ".format(self.env_file))
            raise Exception(e)

        # # get limits max_vms
        # self.max_preemptible = env_json['global_limits']['ec2']['preemptible']
        # self.max_on_demand = env_json['global_limits']['ec2']['on-demand']

        self.env = {}

        for instance in InstanceType.from_dict(env_json):
            # start global count list
            self.count_list[instance.type] = 0

            # logging.info(f"instance_type:{instance}")

            self.env[instance.type] = instance

            self.instances_list.append(instance)

    def __load_loc(self):
        """
        Read the file loc_file and create a dictionary with all available regions
        """

        try:
            with open(self.loc_file) as f:
                loc_json = json.load(f)
        except Exception as e:
            logging.error("<Loader>: Error file {} ".format(self.loc_file))
            raise Exception(e)

        self.loc = {}

        for region in CloudRegion.from_dict(loc_json):
            # logging.info(f"region:{region}")
            self.loc[region.id] = region

    def __get_execution_id(self):
        # """
        # Get next execution_id
        # :return: int
        # """
        # self.execution_id = 1

        """
        Read the database to get the next execution_id
        """

        repo = PostgresRepo()
        row = repo.get_execution(current_filter={'job_id': self.job.job_id, 'limit': 1, 'order': 'desc'})
        if len(row) == 0:
            self.execution_id = 0
        else:
            # get least execution ID
            self.execution_id = row[0].execution_id + 1

    def __update_prices(self):
        """
        get current instances prices on EC2 and GCP and update the env dictionary and also the env.json input file
        """

        # ec2_zone = self.ec2_conf.zone
        # ec2_region = self.ec2_conf.region
        # gcp_zone = self.gcp_conf.zone
        # gcp_region = self.gcp_conf.region

        for instance in self.env.values():

            # print(instance.type)
            if instance.provider == CloudManager.CLOUDLAB:
                for loc in instance.locations:
                    region = loc.split('_')[-1]
                    price = float(instance.vcpu) * float(self.cloudlab_conf.cpu_costs) + float(instance.memory) \
                            * float(self.cloudlab_conf.ram_costs)
                    if instance.gpu == "K40":
                        price = price + float(instance.count_gpu) + float(self.cloudlab_conf.gpu_k40_costs)
                    elif instance.gpu == "P100":
                        price = price + float(instance.count_gpu) + float(self.cloudlab_conf.gpu_p100_costs)
                    elif instance.gpu == "V100":
                        price = price + float(instance.count_gpu) + float(self.cloudlab_conf.gpu_v100_costs)
                    # print(f"Final on-demand price = {price}")
                    instance.setup_ondemand_price(
                        price=price,
                        region=region
                    )
                    price = price * 0.3  # 70% of discount
                    # print(f"Final preemptible price = {price}")
                    instance.setup_preemptible_price(
                        price=price,
                        region=region,
                        zone=''
                    )

                continue

            for loc in self.loc.values():

                region = loc.region
                zone = loc.zones[0]

                if instance.provider == CloudManager.EC2 and loc.provider in (CloudManager.EC2, CloudManager.AWS):
                    logging.error("Without account of EC2 configured")

                    # if instance.market_ondemand:
                    #     instance.setup_ondemand_price(
                    #         price=EC2Manager.get_ondemand_price(instance_type=instance.type, region=region),
                    #         region=region
                    #     )
                    #     # print(f"Final On-demand price: ", instance.price_ondemand)
                    #
                    # if instance.market_preemptible:
                    #     instance.setup_preemptible_price(
                    #         price=EC2Manager.get_preemptible_price(instance_type=instance.type, zone=zone,
                    #                                                region=region)[0][1],
                    #         region=region,
                    #         zone=zone
                    #     )
                    #     # print(f"Final Preemptible price: ", instance.price_preemptible)

                elif instance.provider in (CloudManager.GCLOUD, CloudManager.GCP) \
                        and loc.provider in (CloudManager.GCLOUD, CloudManager.GCP):

                    if instance.market_ondemand:
                        cpu_price, mem_price = GCPManager.get_ondemand_price(instance_type=instance.type,
                                                                             region=region)
                        gpu_price = 0.0
                        if instance.have_gpu:
                            # print("On-demand Price without GPU: ", instance.vcpu * cpu_price + instance.memory *
                            #       mem_price)
                            gpu_price = GCPManager.get_ondemand_gpu_price(gpu_type=instance.gpu, region=region)
                        price = instance.vcpu * cpu_price + instance.memory * mem_price + instance.count_gpu * gpu_price
                        # print(f"Final On-demand price: ", instance.price_ondemand)
                        instance.setup_ondemand_price(
                            price=price,
                            region=region
                        )

                    if instance.market_preemptible:
                        cpu_price, mem_price = GCPManager.get_preemptible_price(instance_type=instance.type,
                                                                                region=region)
                        gpu_price = 0.0
                        if instance.have_gpu:
                            # print("Preemptible Price without GPU: ",
                            #       instance.vcpu * cpu_price + instance.memory * mem_price)
                            gpu_price = GCPManager.get_preemptible_gpu_price(gpu_type=instance.gpu, region=region)
                        price = instance.vcpu * cpu_price + instance.memory * mem_price + instance.count_gpu * gpu_price
                        # print(f"Final Preemptible price: ", instance.price_preemptible)
                        instance.setup_preemptible_price(
                            price=price,
                            region=region,
                            zone=zone
                        )

        # Update env file
        with open(self.env_file, "r") as jsonFile:
            data = json.load(jsonFile)

        # updating prices on env_file
        tmp = data["instances"]
        for instance_type in tmp:
            tmp[instance_type]['prices']['on-demand'] = self.env[instance_type].price_ondemand
            tmp[instance_type]['prices']['preemptible'] = self.env[instance_type].price_preemptible

        data["instances"] = tmp

        with open(self.env_file, "w") as jsonFile:
            json.dump(data, jsonFile, sort_keys=False, indent=4, default=str)

    def __update_zones(self):
        """
        get current zones within regions on EC2 and GCP and update the loc dictionary and also the loc.json input file
        """
        gcp_manager = GCPManager()

        for region in self.loc.values():

            if region.provider == CloudManager.EC2 or region.provider == CloudManager.AWS:
                logging.error("Without account of EC2 configured")
                # region.setup_zones(zones=EC2Manager.get_availability_zones(region.region))

            elif region.provider == CloudManager.GCLOUD or region.provider == CloudManager.GCP:
                region.setup_zones(zones=gcp_manager.get_availability_zones(region.region))

        del gcp_manager

        # Update loc file
        with open(self.loc_file, "r") as jsonFile:
            data = json.load(jsonFile)

        # updating prices on env_file
        tmp = data["locations"]
        for region in tmp:
            tmp[region]['zones'] = self.loc[region].zones

        data["locations"] = tmp

        with open(self.loc_file, "w") as jsonFile:
            json.dump(data, jsonFile, sort_keys=False, indent=4, default=str)

    # update Federated Learning command
    def __update_command(self, strategy, num_seed):
        if self.application_conf.fl_framework == 'flower':
            if strategy is not None:
                self.job.server_task.command = "{0} --rounds {1} --sample_fraction 1 --min_sample_size {2}" \
                                              " --min_num_clients {2} --server_address [::]:{3} --strategy {4}"\
                    .format(self.job.server_task.simple_command,
                            self.job.server_task.n_rounds,
                            self.job.server_task.n_clients,
                            self.application_conf.fl_port,
                            strategy
                            )
                if strategy == 'FedAvgSave':
                    self.job.server_task.command = self.job.server_task.command + f" --frequency_ckpt " \
                                                                                  f"{self.frequency_ckpt}"
            else:
                self.job.server_task.command = "{0} --rounds {1} --sample_fraction 1 --min_sample_size {2}" \
                                              " --min_num_clients {2} --server_address [::]:{3}"\
                    .format(self.job.server_task.simple_command,
                            self.job.server_task.n_rounds,
                            self.job.server_task.n_clients,
                            self.application_conf.fl_port
                            )
            print("server command", self.job.server_task.command)
            for i in range(self.job.num_clients):
                if self.job.client_tasks[i].trainset_dir is not None and self.job.client_tasks[i].trainset_dir != "":
                    predst = os.path.join(self.file_system_conf.path_storage,
                                          self.job.client_tasks[i].trainset_dir)
                else:
                    predst = os.path.join(self.file_system_conf.path_storage)
                if self.job.client_tasks[i].test_dir is not None and self.job.client_tasks[i].test_dir != "":
                    if num_seed is not None:
                        self.job.client_tasks[i].command = "{0} -predst {1} -split {2} -b {3} -out {4} -wpath {5} " \
                                                           "-model_dir {5} -logdir {5} -cache {5} -test_dir {6} " \
                                                           " -epochs {7} -num_seed {8}"\
                            .format(self.job.client_tasks[i].simple_command,
                                    predst,
                                    self.job.client_tasks[i].split,
                                    self.job.client_tasks[i].batch,
                                    os.path.join(self.file_system_conf.path, 'logs'),
                                    os.path.join(self.file_system_conf.path, 'results'),
                                    os.path.join(self.file_system_conf.path_storage,
                                                 self.job.client_tasks[i].test_dir),
                                    self.job.client_tasks[i].train_epochs,
                                    num_seed
                                    )
                    else:
                        self.job.client_tasks[i].command = "{0} -predst {1} -split {2} -b {3} -out {4} -wpath {5} " \
                                                           "-model_dir {5} -logdir {5} -cache {5} -test_dir {6} " \
                                                           " -epochs {7}" \
                            .format(self.job.client_tasks[i].simple_command,
                                    predst,
                                    self.job.client_tasks[i].split,
                                    self.job.client_tasks[i].batch,
                                    os.path.join(self.file_system_conf.path, 'logs'),
                                    os.path.join(self.file_system_conf.path, 'results'),
                                    os.path.join(self.file_system_conf.path_storage,
                                                 self.job.client_tasks[i].test_dir),
                                    self.job.client_tasks[i].train_epochs
                                    )
                else:
                    self.job.client_tasks[i].command = "{0} -predst {1} -split {2} -b {3} -out {4} -wpath {5} " \
                                                      "-model_dir {5} -logdir {5} -cache {5} -epochs {6} " \
                        .format(self.job.client_tasks[i].simple_command,
                                predst,
                                self.job.client_tasks[i].split,
                                self.job.client_tasks[i].batch,
                                os.path.join(self.file_system_conf.path, 'logs'),
                                os.path.join(self.file_system_conf.path, 'results'),
                                self.job.client_tasks[i].train_epochs
                                )
                if self.checkpoint_conf.client_checkpoint:
                    self.job.client_tasks[i].command = "{0} --save_ckpt --restore_ckpt -ckpt_file {1}" \
                        .format(
                        self.job.client_tasks[i].command,
                        self.checkpoint_conf.ckpt_file
                    )

    def print_execution_info(self):
        logging.info("\n")

        header_msg = 25 * "#" + "    Control-GPU {}    ".format(Loader.VERSION) + 25 * "#"
        logging.info(header_msg)
        logging.info("")

        # logging.info("\tExecuting type: '{}' Scheduler: '{}'".format(self.client_command, self.scheduler_name))
        # logging.info("")
        logging.info("\tExecuting type: '{}'".format(self.client_command))
        logging.info("")
        logging.info("\tInput Files:")
        logging.info("\tJob: {}".format(self.job_file))
        logging.info("\tEnv: {}".format(self.env_file))
        logging.info("\tLoc: {}".format(self.loc_file))
        logging.info("\tPre: {}".format(self.pre_file))
        # logging.info("\tMap: {}".format(self.map_file))
        logging.info("\tLog File: {}".format(self.log_file))
        logging.info("\tDaemon AWS: {}".format(self.daemon_aws_file))
        logging.info("\tDaemon GCP: {}".format(self.daemon_gcp_file))
        logging.info("")
        logging.info("")
        logging.info("\t" + 30 * "*")
        logging.info("\tJob id: {} Execution id: {}".format(self.job.job_id, self.execution_id))
        logging.info("\tCommand: {}".format(self.job.server_task.command))
        logging.info("\tNum. Clients: {}".format(self.job.num_clients))
        for i in range(self.job.num_clients):
            logging.info("\tClient {} Task id: {}".format(i, self.job.client_tasks[i].task_id))
            logging.info("\tCommand: {}".format(self.job.client_tasks[i].command))
        logging.info("\tDeadline: {}".format(self.deadline_seconds))
        logging.info("\t" + 30 * "*")

        logging.info("")
        logging.info("\tWITH SIMULATION: {}".format(self.simulation_conf.with_simulation))
        if self.simulation_conf.with_simulation:
            logging.info("\tRevocation Rate: {}".format(self.revocation_rate))
        logging.info("\tWITH CHECKPOINT: {}".format(self.checkpoint_conf.with_checkpoint))
        logging.info("\tServer Checkpoint: {}".format(self.checkpoint_conf.server_checkpoint))
        logging.info("\tClient Checkpoint: {}".format(self.checkpoint_conf.client_checkpoint))
        logging.info("\tDEBUG MODE: {}".format(self.debug_conf.debug_mode))
        logging.info("")

        logging.info(len(header_msg) * "#")
        logging.info("\n\n")
