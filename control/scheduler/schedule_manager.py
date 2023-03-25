import json

import time

from datetime import timedelta, datetime

import logging
from typing import List

from zope.event import subscribers

from control.domain.task import Task
from control.domain.job import Job

from control.managers.virtual_machine import VirtualMachine
from control.managers.dispatcher import Dispatcher
from control.managers.cloud_manager import CloudManager
from control.managers.ec2_manager import EC2Manager
from control.managers.experiment_cloudlab import Experiment

from control.simulators.status_simulator import RevocationSim

from control.repository.postgres_repo import PostgresRepo
from control.repository.postgres_objects import Job as JobRepo
from control.repository.postgres_objects import Task as TaskRepo
from control.repository.postgres_objects import InstanceType as InstanceTypeRepo
from control.repository.postgres_objects import Statistic as StatisticRepo
from control.repository.postgres_objects import Instance as InstanceRepo

from control.scheduler.fl_simple_scheduler import FLSimpleScheduler
from control.scheduler.mathematical_formulation_scheduler import MathematicalFormulationScheduler
from control.scheduler.dynamic_scheduler import DynamicScheduler
from control.scheduler.scheduler import Scheduler

from control.util.loader import Loader

import threading

import os


class ScheduleManager:

    # hibernated_dispatcher: List[Dispatcher]
    terminated_dispatcher: List[Dispatcher]
    idle_dispatchers: List[Dispatcher]
    working_dispatchers: List[Dispatcher]
    extra_vm: VirtualMachine
    # hibernating_dispatcher: List[Dispatcher]
    job_status = Task.WAITING

    def __init__(self, loader: Loader):

        self.loader = loader

        # load the Scheduler that will be used
        self.__load_scheduler()
        # print(self.scheduler)

        # read expected_makespan on build_dispatcher()
        # self.expected_makespan_seconds = None
        # self.deadline_timestamp = None

        '''
           If the execution has simulation
           Prepare the simulation environment
        '''
        if self.loader.simulation_conf.with_simulation:
            # start simulator
            self.simulator = RevocationSim(self.loader.revocation_rate)

        # Keep Used EBS Volumes
        self.ebs_volumes = []

        # Vars Datetime to keep track of global execution time
        self.start_timestamp = None
        self.end_timestamp = None
        self.elapsed_time = None

        self.repo = PostgresRepo()

        # Semaphore
        self.semaphore = threading.Semaphore()
        # self.semaphore_count = threading.Semaphore()

        # TRACKERS VALUES
        self.n_interruptions = 0
        self.n_sim_interruptions = 0

        self.timeout = False

        ''' ABORT FLAG'''
        self.abort = False

        self.server_task_dispatcher: Dispatcher
        self.client_tasks_dispatchers: List[Dispatcher]
        self.extra_vm = None
        self.terminated_dispatchers = []
        self.working_dispatchers = []
        self.idle_dispatchers = []

        self.server_revoked = False
        # self.server_task_status = Task.WAITING
        # self.client_tasks_status = []
        # for i in range(self.loader.job.num_clients):
        #     self.client_tasks_status.append(Task.WAITING)

        # Prepare the control database and the folders structure in S3
        try:
            self.__prepare_execution()
        except Exception as e:
            logging.error(e)
            raise e

        '''
                Build the initial dispatchers
                The class Dispatcher is responsible to manager the execution steps
                '''
        self.__build_dispatchers()

    # # PRE-EXECUTION FUNCTIONS

    def __load_scheduler(self):

        if self.loader.scheduler_name.upper() == Scheduler.FL_SIMPLE:
            if self.loader.server_provider is None:
                logging.error("<Loader>: Server provider cannot be None")
                return
            if self.loader.server_region is None:
                logging.error("<Loader>: Server region cannot be None")
                return
            if self.loader.server_vm_name is None:
                logging.error("<Loader>: Server VM name cannot be None")
                return
            if self.loader.clients_provider is None:
                logging.error("<Loader>: Clients provider cannot be None")
                return
            if self.loader.clients_region is None:
                logging.error("<Loader>: Clients region cannot be None")
                return
            if self.loader.clients_vm_name is None:
                logging.error("<Loader>: Clients VM name cannot be None")
                return
            self.scheduler = FLSimpleScheduler(instance_types=self.loader.env, locations=self.loader.loc)

        elif self.loader.scheduler_name.upper() == Scheduler.MAT_FORM:
            self.scheduler = MathematicalFormulationScheduler(loader=self.loader)

        elif self.loader.scheduler_name.upper() == Scheduler.DYNAMIC_GREEDY:
            self.scheduler = DynamicScheduler(loader=self.loader)

        if self.scheduler is None:
            logging.error("<Scheduler Manager {}_{}>: "
                          "ERROR - Scheduler {} not found".format(self.loader.job.job_id,
                                                                  self.loader.execution_id,
                                                                  self.loader.scheduler_name))
            Exception("<Scheduler Manager {}_{}>:  "
                      "ERROR - Scheduler {} not found".format(self.loader.job.job_id,
                                                              self.loader.execution_id,
                                                              self.loader.scheduler_name))

    def __build_dispatchers(self):
        try:
            with open(self.loader.map_file) as f:
                data = f.read()
            json_data = json.loads(data)
        except Exception:
            json_data = None
        if json_data is None:
            aux_provider = self.loader.server_provider,
            aux_region = self.loader.server_region,
            aux_vm_name = self.loader.server_vm_name
        else:
            aux_provider = json_data['server']['provider']
            aux_region = json_data['server']['region']
            aux_vm_name = json_data['server']['instance_type']

        # logging.info("Starts building dispatchers")
        instance_type, market, region, zone = self.scheduler.get_server_instance(
            provider=aux_provider,
            region=aux_region,
            vm_name=aux_vm_name
        )
        # Create the Vm that will be used by the dispatcher
        vm = VirtualMachine(
            instance_type=instance_type,
            market=market,
            loader=self.loader,
            region=region,
            zone=zone,
            simulator=self.simulator
        )

        # Then a dispatcher, that will execute the tasks, is created

        server_dispatcher = Dispatcher(vm=vm, loader=self.loader,
                                       type_task=Job.SERVER, client_id=self.loader.job.num_clients)

        # check if the VM need to be registered on the simulator
        if self.loader.simulation_conf.with_simulation and vm.market in (CloudManager.PREEMPTIBLE, Experiment.MARKET)\
                and self.loader.simulation_conf.faulty_server and not self.loader.emulated:
            logging.info("<Scheduler Manager {}_{}>: Revogation simulation of server".format(self.loader.job.job_id,
                                                                                             self.loader.execution_id))
            self.simulator.register_vm(vm)

        self.server_task_dispatcher = server_dispatcher

        self.working_dispatchers.append(server_dispatcher)

        self.client_task_dispatchers = []

        for i in range(self.loader.job.num_clients):
            # client = self.loader.job.client_tasks[i]
            if json_data is None:
                aux_provider = self.loader.clients_provider[i],
                aux_region = self.loader.clients_region[i],
                aux_vm_name = self.loader.clients_vm_name[i]
            else:
                aux_provider = json_data['clients'][str(i)]['provider']
                aux_region = json_data['clients'][str(i)]['region']
                aux_vm_name = json_data['clients'][str(i)]['instance_type']

            instance_type, market, region, zone = self.scheduler.get_client_instance(
                provider=aux_provider,
                region=aux_region,
                vm_name=aux_vm_name,
                client_id=i
            )

            # Create the Vm that will be used by the dispatcher
            vm = VirtualMachine(
                instance_type=instance_type,
                market=market,
                loader=self.loader,
                region=region,
                zone=zone,
                simulator=self.simulator
            )

            # Then a dispatcher, that will execute the tasks, is created
            client_dispatcher = Dispatcher(vm=vm, loader=self.loader,
                                           type_task=Job.CLIENT, client_id=i)

            # check if the VM need to be registered on the simulator
            if self.loader.simulation_conf.with_simulation and vm.market in (CloudManager.PREEMPTIBLE,
                                                                             Experiment.MARKET) and \
                    self.loader.simulation_conf.faulty_clients:
                logging.info("<Scheduler Manager {}_{}>: Revogation simulation "
                             "of client {}".format(self.loader.job.job_id,
                                                   self.loader.execution_id,
                                                   i))
                if not self.loader.emulated:
                    self.simulator.register_vm(vm)

            self.semaphore.acquire()

            self.client_task_dispatchers.append(client_dispatcher)
            self.working_dispatchers.append(client_dispatcher)

            self.semaphore.release()

    def __prepare_execution(self):
        """
           Prepare control database and all directories to start the execution process
        """
        # get job from control database
        jobs_repo = self.repo.get_jobs(
            current_filter={
                'job_id': self.loader.job.job_id
            }
        )

        # Check if Job is already in the database
        if len(jobs_repo) == 0:
            # add job to database
            self.__add_job_to_database()
        else:
            # Job is already in database
            # Check job and Instances consistency
            logging.info("<Scheduler Manager {}_{}>: - "
                         "Checking database consistency...".format(self.loader.job.job_id,
                                                                   self.loader.execution_id))

            job_repo = jobs_repo[0]

            assert job_repo.name == self.loader.job.job_name, "Consistency error (job name): {} <> {}".format(
                job_repo.name, self.loader.job.job_name)

            assert job_repo.description == self.loader.job.description, "Consistency error (job description): " \
                                                                        "{} <> {} ".format(job_repo.description,
                                                                                           self.loader.job.description)

            tasks_repo = job_repo.tasks.all()

            assert len(tasks_repo) == \
                   self.loader.job.total_tasks, "Consistency error (number of tasks): {} <> {} ".format(
                len(tasks_repo), self.loader.job.total_tasks
            )

            # check tasks consistency
            for t in tasks_repo:
                # server task
                if 'server' in t.command:
                    task = self.loader.job.server_task

                    assert t.task_id == task.task_id, \
                        "Consistency error (server task id): {} <>".format(t.task_id,
                                                                           self.loader.job.server_task.task_id)

                    assert task.task_name == t.task_name, "Consistency error (server task {} memory): " \
                                                          "{} <> {} ".format(task.task_id, t.task_name, task.task_name)

                    assert task.simple_command == t.command, "Consistency error (server task {} command): " \
                                                             "{} <> {} ".format(task.task_id, t.command,
                                                                                task.simple_command)

                # client tasks
                else:
                    assert t.task_id in self.loader.job.client_tasks, "Consistency error (client task id): {}".format(
                        t.task_id)

                    task = self.loader.job.client_tasks[t.task_id]

                    assert task.task_name == t.task_name, "Consistency error (client task {} memory): " \
                                                          "{} <> {} ".format(task.task_id, t.task_name, task.task_name)

                    assert task.simple_command == t.command, "Consistency error (client task {} command): " \
                                                             "{} <> {} ".format(task.task_id, t.command,
                                                                                task.simple_command)

        # Check Instances Type
        for key, instance_type in self.loader.env.items():

            types = self.repo.get_instance_type(current_filter={
                'instance_type': key
            })

            if len(types) == 0:
                # add instance to control database
                self.__add_instance_type_to_database(instance_type)
            else:
                # check instance type consistency
                inst_type_repo = types[0]
                assert inst_type_repo.vcpu == instance_type.vcpu, "Consistency error (vcpu instance {}): " \
                                                                  "{} <> {} ".format(key,
                                                                                     inst_type_repo.vcpu,
                                                                                     instance_type.vcpu)

                assert inst_type_repo.memory == instance_type.memory, "Consistency error (memory instance {}):" \
                                                                      "{} <> {}".format(key,
                                                                                        inst_type_repo.memory,
                                                                                        instance_type.memory)

    def __add_job_to_database(self):
        """Record a Job and its tasks to the control database"""

        job_repo = JobRepo(
            id=self.loader.job.job_id,
            name=self.loader.job.job_name,
            description=self.loader.job.description
        )

        self.repo.add_job(job_repo)

        # add tasks
        self.repo.add_task(
            TaskRepo(
                job=job_repo,
                task_id=self.loader.job.server_task.task_id,
                task_name=self.loader.job.server_task.task_name,
                command=self.loader.job.server_task.simple_command
            )
        )
        for task_id, task in self.loader.job.client_tasks.items():
            self.repo.add_task(
                TaskRepo(
                    job=job_repo,
                    task_id=task.task_id,
                    task_name=task.task_name,
                    command=task.simple_command,
                )
            )

    def __add_instance_type_to_database(self, instance_type):
        self.repo.add_instance_type(
            InstanceTypeRepo(
                type=instance_type.type,
                provider=instance_type.provider,
                vcpu=instance_type.vcpu,
                memory=instance_type.memory
            )
        )

    '''
    HANDLES FUNCTIONS
    '''

    def __idle_handle(self, dispatcher: Dispatcher, type_affected_task, affected_client_id):
        self.semaphore.acquire()

        if dispatcher in self.working_dispatchers:
            self.working_dispatchers.remove(dispatcher)

            self.idle_dispatchers.append(dispatcher)

        self.semaphore.release()

        if type_affected_task == Job.SERVER:
            self.loader.job.server_task.finish_execution()
            # self.server_task_status = Task.FINISHED
        elif affected_client_id in self.loader.job.client_tasks:
            self.loader.job.client_tasks[affected_client_id].finish_execution()

    def __interruption_handle(self):
        pass
        # # Move task to other VM
        # # self.semaphore.acquire()
        #
        # if not self.loader.cudalign_task.has_task_finished():
        #     self.loader.cudalign_task.stop_execution()
        #
        # # logging.info("Entered interruption_handle")
        #
        # # getting volume-id
        # if self.loader.file_system_conf.type == EC2Manager.EBS:
        #     self.ebs_volume_id = self.task_dispatcher.vm.volume_id
        #
        # # logging.info("Got EBS id: {}".format(self.ebs_volume_id))
        #
        # # See in which VM we will restart
        # current_time = self.start_timestamp - datetime.now()
        #
        # instance_type, market = self.scheduler.choose_restart_best_instance_type(
        #     cudalign_task=self.loader.cudalign_task,
        #     deadline=self.loader.deadline_seconds,
        #     current_time=current_time.total_seconds()
        # )
        #
        # # logging.info("Chose instance {} of type {}".format(instance_type.type, market))
        #
        # if self.loader.cudalign_task.has_task_finished():
        #     new_vm = VirtualMachine(
        #         instance_type=instance_type,
        #         market=market,
        #         loader=self.loader,
        #         volume_id=self.ebs_volume_id
        #     )
        #
        #     # logging.info("Created new VM!")
        #
        #     dispatcher = Dispatcher(vm=new_vm, loader=self.loader)
        #
        #     # check if the VM need to be registered on the simulator
        #     if self.loader.simulation_conf.with_simulation and new_vm.market == CloudManager.PREEMPTIBLE:
        #         self.simulator.register_vm(new_vm)
        #
        #     # self.semaphore.acquire()
        #
        #     self.terminated_dispatchers.append(self.task_dispatcher)
        #     self.task_dispatcher = dispatcher
        #
        #     # self.semaphore.release()
        #
        #     self.__start_dispatcher()
        #
        # # self.semaphore.release()

    def __terminated_handle(self, affected_dispatcher: Dispatcher, terminate_vm):
        # Move task to other VM
        # # self.semaphore.acquire()
        #
        if affected_dispatcher.type_task == Job.SERVER:
            self.server_revoked = True
            self.semaphore.acquire()

            if affected_dispatcher in self.working_dispatchers:
                self.working_dispatchers.remove(affected_dispatcher)

                self.terminated_dispatchers.append(affected_dispatcher)

            # if len(self.working_dispatchers) == 1:
            #     self.abort = True

            self.semaphore.release()

            if terminate_vm:
                affected_dispatcher.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

            if not self.loader.job.server_task.has_task_finished():
                self.loader.job.server_task.stop_execution()

            logging.info("Entered terminated_handle to server VM")

            new_provider, new_region, new_vm_name = self.scheduler.choose_server_new_instance()

            instance_type, market, region, zone = self.scheduler.get_server_instance(
                provider=new_provider,
                region=new_region,
                vm_name=new_vm_name
            )

            logging.info("Chosen instance {} of type {} in region {} in zone {}".format(instance_type.type, market,
                                                                                        region, zone))

            if not self.loader.job.server_task.has_task_finished():

                # Create the Vm that will be used by the dispatcher
                new_vm = VirtualMachine(
                    instance_type=instance_type,
                    market=market,
                    loader=self.loader,
                    region=region,
                    zone=zone,
                    simulator=self.simulator
                )

                new_dispatcher = Dispatcher(
                    vm=new_vm,
                    loader=self.loader,
                    type_task=Job.SERVER,
                    client_id=self.loader.job.num_clients,
                    needs_external_transfer=True
                )

                # check if the VM need to be registered on the simulator
                if self.loader.simulation_conf.with_simulation and new_vm.market == CloudManager.PREEMPTIBLE:
                    self.simulator.register_vm(new_vm)

                self.semaphore.acquire()

                self.working_dispatchers.append(new_dispatcher)
                self.server_task_dispatcher = new_dispatcher

                new_dispatcher.main_thread.start()
                time.sleep(60)
                while not self.server_task_dispatcher.vm.ready:
                    if self.server_task_dispatcher.vm.failed_to_created:
                        break
                    time.sleep(1)
                    # print("Testing vm.ready: ", self.server_task_dispatcher.vm.ready)

                if not self.server_task_dispatcher.vm.failed_to_created:

                    # update number of rounds
                    server_ckpt_round = 0
                    client_ckpt_round = 0
                    if self.loader.checkpoint_conf.server_checkpoint:

                        self.extra_vm.open_connection()

                        if not self.extra_vm.ssh.is_active:
                            self.extra_vm.ssh.open_connection()
                        try:
                            stdout, stderr, code_return = self.extra_vm.ssh.execute_command(
                                "ls {} | grep '.npz$'".format(self.loader.checkpoint_conf.folder_checkpoints),
                                output=True)
                            print("stdout", stdout)
                            print("stderr", stderr)
                            print("code_return", code_return)
                            aux_list = stdout.split('\n')
                            # print("aux_list", aux_list)
                            last_line = aux_list[-2].split('-')
                            # print("last_line", last_line)
                            server_ckpt_round = int(last_line[1])
                        except Exception as e:
                            logging.error(e)
                            server_ckpt_round = 0

                        self.server_task_dispatcher.vm.prepare_ft_daemon(ip_address=self.extra_vm.instance_public_ip,
                                                                         restart=True)

                    if self.loader.checkpoint_conf.client_checkpoint:
                        for dispatcher in self.client_task_dispatchers:
                            try:
                                dispatcher.vm.open_connection()

                                if not dispatcher.vm.ssh.is_active:
                                    dispatcher.vm.ssh.open_connection()
                                stdout, stderr, code_return = dispatcher.vm.ssh.execute_command(
                                    f"cat {self.loader.checkpoint_conf.ckpt_file}",
                                    output=True)
                                print("stdout", stdout)
                                print("stderr", stderr)
                                print("code_return", code_return)
                                aux_list = stdout.split('\n')
                                # print("aux_list", aux_list)
                                last_line = aux_list[-1].split('-')
                                # print("last_line", last_line)
                                client_ckpt_round = int(last_line[1])
                                break
                            except Exception as e:
                                logging.error(e)
                                client_ckpt_round = 0
                                continue

                    logging.info(f"client_ckpt_round {client_ckpt_round} - server_ckpt_round {server_ckpt_round}")

                    folder_ckpt = self.loader.checkpoint_conf.folder_checkpoints
                    if folder_ckpt[-1] == '/':
                        folder_ckpt = folder_ckpt[:-1]

                    self.server_task_dispatcher.vm.open_connection()

                    if not self.server_task_dispatcher.vm.ssh.is_active:
                        self.server_task_dispatcher.vm.ssh.open_connection()

                    if client_ckpt_round >= server_ckpt_round:
                        if self.loader.checkpoint_conf.server_checkpoint:
                            self.extra_vm.ssh.execute_command(f'rm {folder_ckpt} -r')
                            self.extra_vm.ssh.execute_command(f'mkdir {folder_ckpt}')

                        self.server_task_dispatcher.vm.ssh.execute_command(f"echo '' > name_checkpoint.txt")

                        # Restarts from client checkpoint
                        self.server_task_dispatcher.update_rounds(client_ckpt_round)
                    else:
                        try:
                            # Restarts from server checkpoint
                            ckpt_file = f"round-{server_ckpt_round}-weights.npz"

                            self.server_task_dispatcher.vm.ssh.execute_command(f"echo {ckpt_file} > "
                                                                               f"name_checkpoint.txt")

                            self.server_task_dispatcher.update_rounds(server_ckpt_round, ckpt_restore=True)
                        except Exception as e:
                            logging.error(e)

                    self.server_task_dispatcher.start_execution = True
                    time.sleep(20)

                    server_ip = f"{self.server_task_dispatcher.vm.instance_public_ip}:" \
                                f"{self.loader.application_conf.fl_port}"
                    print("new server_ip:", server_ip)
                    for i in range(self.loader.job.num_clients):
                        self.loader.job.client_tasks[i].server_ip = server_ip
                    for working_dispatcher in self.working_dispatchers:
                        working_dispatcher.update_server_ip(server_ip)
                        # working_dispatcher.main_thread.start()

                self.semaphore.release()
                self.server_revoked = False

        elif affected_dispatcher.type_task == Job.CLIENT and not self.server_revoked:
            self.semaphore.acquire()

            if affected_dispatcher in self.working_dispatchers:
                self.working_dispatchers.remove(affected_dispatcher)

                self.terminated_dispatchers.append(affected_dispatcher)

            # if len(self.working_dispatchers) == 1:
            #     self.abort = True

            self.semaphore.release()

            if terminate_vm:
                affected_dispatcher.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

            if not self.loader.job.client_tasks[affected_dispatcher.client.client_id].has_task_finished():
                self.loader.job.client_tasks[affected_dispatcher.client_id].stop_execution()

            logging.info("Entered terminated_handle to client VM")

            new_provider, new_region, new_vm_name = self.scheduler.choose_client_new_instance(
                affected_dispatcher.client_id
            )

            instance_type, market, region, zone = self.scheduler.get_client_instance(
                provider=new_provider,
                region=new_region,
                vm_name=new_vm_name,
                client_id=affected_dispatcher.client_id
            )

            logging.info("Chosen instance {} of type {} in region {} in zone {}".format(instance_type.type, market,
                                                                                        region, zone))

            if not self.loader.job.client_tasks[affected_dispatcher.client.client_id].has_task_finished():

                # Create the Vm that will be used by the dispatcher
                new_vm = VirtualMachine(
                    instance_type=instance_type,
                    market=market,
                    loader=self.loader,
                    region=region,
                    zone=zone,
                    simulator=self.simulator
                )

                new_dispatcher = Dispatcher(
                    vm=new_vm,
                    loader=self.loader,
                    client_id=affected_dispatcher.client_id,
                    type_task=Job.CLIENT
                )

                # check if the VM need to be registered on the simulator
                if self.loader.simulation_conf.with_simulation and new_vm.market == CloudManager.PREEMPTIBLE:
                    self.simulator.register_vm(new_vm)

                self.semaphore.acquire()

                self.working_dispatchers.append(new_dispatcher)
                self.client_task_dispatchers[affected_dispatcher.client_id] = new_dispatcher

                new_dispatcher.main_thread.start()
                time.sleep(60)

                self.semaphore.release()

        #
        # if not self.loader.cudalign_task.has_task_finished():
        #     new_vm = VirtualMachine(
        #         instance_type=instance_type,
        #         market=market,
        #         loader=self.loader,
        #         volume_id=self.ebs_volume_id
        #     )
        #
        #     # logging.info("Created new VM!")
        #
        #     dispatcher = Dispatcher(vm=new_vm, loader=self.loader)
        #
        #     # check if the VM need to be registered on the simulator
        #     if self.loader.simulation_conf.with_simulation and new_vm.market == CloudManager.PREEMPTIBLE:
        #         self.simulator.register_vm(new_vm)
        #
        #     # self.semaphore.acquire()
        #
        #     self.terminated_dispatchers.append(self.task_dispatcher)
        #     self.task_dispatcher = dispatcher
        #
        #     # self.semaphore.release()
        #
        #     self.__start_dispatcher()
        #
        # # self.semaphore.release()

    def __event_handle(self, event):

        affected_dispatcher: Dispatcher = event.kwargs['dispatcher']
        type_affected_task = event.kwargs['type_task']
        affected_client_id = event.kwargs['client_id']

        logging.info("<Scheduler Manager {}_{}>: - EVENT_HANDLE "
                     "Instance: '{}', Type: '{}', Market: '{}',"
                     "Event: '{}'".format(self.loader.job.job_id,
                                          self.loader.execution_id,
                                          affected_dispatcher.vm.instance_id,
                                          affected_dispatcher.vm.type,
                                          affected_dispatcher.vm.market,
                                          event.value))

        if event.value == CloudManager.IDLE and affected_dispatcher.working:

            if affected_dispatcher.type_task == Job.SERVER or (affected_dispatcher.type_task == Job.CLIENT
                                                               and not self.server_revoked):
                logging.info("<Scheduler Manager {}_{}>: - Calling Idle Handle".format(self.loader.job.job_id,
                                                                                       self.loader.execution_id))
                self.__idle_handle(affected_dispatcher, type_affected_task, affected_client_id)

            # self.client_tasks_status[i] = Task.FINISHED
        # elif event.value == CloudManager.STOPPED:
        #     # self.semaphore_count.acquire()
        #     self.n_interruptions += 1
        #     # self.semaphore_count.release()
        #
        #     self.task_dispatcher.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)
        #
        #     logging.info("<Scheduler Manager {}_{}>: - Calling Interruption Handle"
        #                  .format(self.loader.cudalign_task.task_id, self.loader.execution_id))
        #     # self.__interruption_handle()
        #
        elif event.value in [CloudManager.TERMINATED, CloudManager.ERROR, CloudManager.STOPPING]:
            logging.info("<Scheduler Manager {}_{}>: - Calling Terminate Handle"
                         .format(self.loader.job.job_id, self.loader.execution_id))
            if not affected_dispatcher.vm.marked_to_interrupt:
                self.n_sim_interruptions += 1
            if event.value == CloudManager.ERROR:
                self.__terminated_handle(affected_dispatcher, terminate_vm=True)
            else:
                self.__terminated_handle(affected_dispatcher, terminate_vm=False)

        elif event.value in CloudManager.ABORT:
            self.abort = True

    '''
    CHECKERS FUNCTIONS
    '''

    def __checkers(self):
        # Checker loop
        # Checker if all dispatchers have finished the execution
        # while len(self.working_dispatchers) > 0 or len(self.hibernating_dispatcher) > 0:
        while len(self.working_dispatchers) > 0:

            if self.abort:
                break

            # # If new checkers would be created that function have to be updated
            # self.__check_hibernated_dispatchers()
            # self.__check_idle_dispatchers()
            time.sleep(5)

    '''
    Manager Functions
    '''

    def __start_server_dispatcher(self):
        # self.semaphore.acquire()

        # Starting working dispatcher
        self.server_task_dispatcher.main_thread.start()
        # print("Testing vm.ready: ", self.server_task_dispatcher.vm.ready)
        while not self.server_task_dispatcher.vm.ready:
            if self.server_task_dispatcher.vm.failed_to_created:
                break
            time.sleep(1)
            # print("Testing vm.ready: ", self.server_task_dispatcher.vm.ready)
        if not self.server_task_dispatcher.vm.failed_to_created:
            server_ip = f"{self.server_task_dispatcher.vm.instance_public_ip}:{self.loader.application_conf.fl_port}"
            print("server_ip:", server_ip)
            for i in range(self.loader.job.num_clients):
                self.loader.job.client_tasks[i].server_ip = server_ip

        # self.semaphore.release()

        return not self.server_task_dispatcher.vm.failed_to_created

    def __start_clients_dispatchers(self):
        self.semaphore.acquire()

        # Starting working dispatcher
        for i in range(self.loader.job.num_clients):
            self.client_task_dispatchers[i].main_thread.start()
            time.sleep(60)

        self.semaphore.release()

    def __terminate_dispatcher(self):

        if self.loader.debug_conf.debug_mode:
            logging.warning(100 * "#")
            logging.warning("\t<DEBUG MODE>: WAITING COMMAND TO TERMINATE -  PRESS ENTER")
            logging.warning(100 * "#")

            input("")

        logging.info("")
        logging.info("<Scheduler Manager {}_{}>: - Start termination process... "
                     .format(self.loader.job.job_id, self.loader.execution_id))

        # terminate simulation
        if self.loader.simulation_conf.with_simulation:
            self.simulator.stop_simulation()

        self.semaphore.acquire()

        # Terminate all working dispatchers
        logging.info("<Scheduler Manager {}_{}>: - "
                     "Terminating working Dispatcher instances".format(self.loader.job.job_id,
                                                                       self.loader.execution_id))
        for working_dispatcher in self.working_dispatchers[:]:
            working_dispatcher.debug_wait_command = False

            working_dispatcher.working = False

            self.working_dispatchers.remove(working_dispatcher)
            self.terminated_dispatchers.append(working_dispatcher)

        # Terminate all  idle dispatchers
        logging.info("<Scheduler Manager {}_{}>: - Terminating idle instances".format(self.loader.job.job_id,
                                                                                      self.loader.execution_id))
        for idle_dispatcher in self.idle_dispatchers[:]:
            idle_dispatcher.debug_wait_command = False

            idle_dispatcher.working = False

            if not idle_dispatcher.main_thread.is_alive():
                idle_dispatcher.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

            self.idle_dispatchers.remove(idle_dispatcher)
            self.terminated_dispatchers.append(idle_dispatcher)

        # Confirm Termination
        logging.info("<Scheduler Manager {}_{}>: - Waiting Termination process...".format(self.loader.job.job_id,
                                                                                          self.loader.execution_id))

        for terminated_dispatcher in self.terminated_dispatchers:
            terminated_dispatcher.debug_wait_command = False
            # waiting thread to terminate

            if terminated_dispatcher.main_thread.is_alive():
                terminated_dispatcher.main_thread.join()

            # getting volume-id
            if self.loader.file_system_conf.type == EC2Manager.EBS:
                self.ebs_volumes.append(terminated_dispatcher.vm.volume_id)

        # if self.extra_vm is not None:
        #     self.extra_vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

        # self.semaphore.release()

    def __end_of_execution(self):

        # end of execution
        self.end_timestamp = datetime.now()
        self.elapsed_time = (self.end_timestamp - self.start_timestamp)

        logging.info("<Scheduler Manager {}_{}>: - Waiting Termination...".format(self.loader.job.job_id,
                                                                                  self.loader.execution_id))

        cost = 0.0
        on_demand_count = 0
        preemptible_count = 0

        for dispatcher in self.terminated_dispatchers:
            if not dispatcher.vm.failed_to_created:

                if dispatcher.vm.market == CloudManager.ON_DEMAND:
                    on_demand_count += 1
                else:
                    preemptible_count += 1

                cost += dispatcher.vm.uptime.seconds * \
                    (dispatcher.vm.price / 3600.0)  # price in seconds'

        logging.info("")

        if not self.abort:
            execution_info = "    Job: {} Execution: {} Scheduler: FLSimpleScheduler    "\
                .format(self.loader.job.job_id, self.loader.execution_id)
        else:
            execution_info = "    Job: {} Execution: {} Scheduler: FLSimpleScheduler" \
                             " - EXECUTION ABORTED    ".format(self.loader.job.job_id,
                                                               self.loader.execution_id)

        execution_info = 20 * "#" + execution_info + 20 * "#"

        logging.info(execution_info)
        logging.info("")
        total = self.n_sim_interruptions + self.n_interruptions

        logging.info("\t AWS interruption: {} Simulation interruption: {} "
                     "Total interruption: {}".format(self.n_interruptions, self.n_sim_interruptions, total))

        total = on_demand_count + preemptible_count
        logging.info(
            "\t On-demand: {} Preemptible: {} Total: {}".format(on_demand_count,
                                                                preemptible_count,
                                                                total))
        logging.info("")
        logging.info("")
        logging.info("\t Start Time: {}  End Time: {}".format(self.start_timestamp, self.end_timestamp))
        logging.info("\t Elapsed Time: {}".format(self.elapsed_time))
        logging.info("\t Deadline: {}".format(timedelta(seconds=self.loader.deadline_seconds)))
        logging.info("")
        logging.info("")
        logging.info("\t Execution Total Estimated monetary Cost: {}".format(cost))
        logging.info("")

        if self.loader.file_system_conf.type == CloudManager.EBS and not self.loader.file_system_conf.ebs_delete:
            logging.warning("The following EBS VOLUMES will not be deleted by Multi-FedLS: ")
            for volume_id in self.ebs_volumes:
                logging.warning("\t-> {}".format(volume_id))

        logging.info("")
        logging.info(len(execution_info) * "#")

        status = 'success'

        if self.abort:
            status = 'aborted'

        self.repo.add_statistic(
            StatisticRepo(execution_id=self.loader.execution_id,
                          job_id=self.loader.job.job_id,
                          status=status,
                          start=self.start_timestamp,
                          end=self.end_timestamp,
                          deadline=self.loader.deadline_timedelta,
                          cost=cost)
        )

        self.repo.close_session()

        if self.abort:
            error_msg = "<Scheduler Manager {}_{}>: - " \
                        "Check all log-files. Execution Aborted".format(self.loader.job.job_id,
                                                                        self.loader.execution_id)
            logging.error(error_msg)
            raise Exception

    def start_execution(self):
        # subscriber events_handle
        subscribers.append(self.__event_handle)

        self.start_timestamp = datetime.now()
        # UPDATE DATETIME DEADLINE

        logging.info("<Scheduler Manager {}_{}>: - Starting Execution.".format(self.loader.job.job_id,
                                                                               self.loader.execution_id))
        logging.info("")

        self.__start_server_dispatcher()

        self.__start_clients_dispatchers()

        if self.loader.checkpoint_conf.extra_vm:
            time.sleep(600)
            self.__start_extra_vm()

        # Call checkers loop
        self.__checkers()

        if self.loader.emulated:
            self.__get_results()

        self.__terminate_dispatcher()

        self.__end_of_execution()

    def __get_results(self):
        logging.info("<Scheduler Manager {}_{}>: - Getting results from VMS".format(self.loader.job.job_id,
                                                                                    self.loader.execution_id))
        folder_root = f"results/{self.loader.job.job_id}_{self.loader.execution_id}"
        threads_dispatcher = []
        thread = threading.Thread(target=self.__get_result_vm, args=(self.server_task_dispatcher.vm,
                                                                     folder_root,
                                                                     self.semaphore))
        thread.start()
        threads_dispatcher.append(thread)
        for dispatcher in self.client_task_dispatchers:
            thread = threading.Thread(target=self.__get_result_vm, args=(dispatcher.vm,
                                                                         folder_root,
                                                                         self.semaphore))
            thread.start()
            threads_dispatcher.append(thread)

        for thread in threads_dispatcher:
            thread.join()

    def __get_result_vm(self, vm, folder_root, semaphore):
        logging.info("<Scheduler Manager {}_{}>: - Getting results from VM {}".format(self.loader.job.job_id,
                                                                                      self.loader.execution_id,
                                                                                      vm.instance_id))
        folder = f"{folder_root}/{vm.experiment_emulation.experiment_name}"
        semaphore.acquire()
        os.makedirs(folder)
        semaphore.release()
        try:
            logging.info("Opening connection")
            if vm.ssh.is_active or vm.ssh.open_connection():
                logging.info("Connection opened")
                vm.ssh.get_dir(source=self.loader.file_system_conf.path, target=folder)
                vm.ssh.close_connection()
        except Exception as e:
            logging.error(e)

    def __start_extra_vm(self):
        try:
            with open(self.loader.map_file) as f:
                data = f.read()
            json_data = json.loads(data)
        except Exception:
            json_data = None
        if json_data is None:
            aux_provider = self.loader.server_provider,
            aux_region = self.loader.server_region
        else:
            aux_provider = json_data['server']['provider']
            aux_region = json_data['server']['region']
        while True:
            instance_type, market, region, zone = self.scheduler.get_extra_vm_instance(
                provider=aux_provider,
                region=aux_region
            )
            # Create the Vm that will be used by the dispatcher
            self.extra_vm = VirtualMachine(
                instance_type=instance_type,
                market=market,
                loader=self.loader,
                region=region,
                zone=zone
            )

            logging.info("Deploying extra VM in {}".format(self.extra_vm.zone))
            logging.info("Vms region {} and image_id {}".format(self.extra_vm.region.region,
                                                                self.extra_vm.region.server_image_id))

            status = self.extra_vm.deploy(type_task='extra_vm')

            if not status:
                for zone in self.extra_vm.region.zones:
                    self.extra_vm.instance_id = None
                    self.extra_vm.zone = zone
                    self.extra_vm.instance_type.zone = zone
                    logging.info("Deploying extra VM in {}".format(self.extra_vm.zone))
                    logging.info("Vms region {} and image_id {}".format(self.extra_vm.region.region,
                                                                        self.extra_vm.region.server_image_id))
                    status = self.extra_vm.deploy(type_task='extra_vm')
                    if status:
                        break

            if status:
                self.repo.add_instance(InstanceRepo(id=f'{self.extra_vm.instance_id}_{self.loader.execution_id}',
                                                    type=self.extra_vm.instance_type.type,
                                                    region=self.extra_vm.instance_type.region,
                                                    zone=self.extra_vm.instance_type.zone,
                                                    market=self.extra_vm.market,
                                                    ebs_volume=self.extra_vm.volume_id,
                                                    price=self.extra_vm.price))

                self.server_task_dispatcher.vm.prepare_ft_daemon(ip_address=self.extra_vm.instance_public_ip)

                break
