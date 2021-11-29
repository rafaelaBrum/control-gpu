# import json

import time

from datetime import timedelta, datetime

import logging
from typing import List

from zope.event import subscribers

from control.domain.task import Task

from control.managers.virtual_machine import VirtualMachine
from control.managers.dispatcher import Dispatcher
from control.managers.cloud_manager import CloudManager
from control.managers.ec2_manager import EC2Manager

from control.simulators.status_simulator import RevocationSim

from control.repository.postgres_repo import PostgresRepo
from control.repository.postgres_objects import Job as JobRepo
from control.repository.postgres_objects import Task as TaskRepo
from control.repository.postgres_objects import InstanceType as InstanceTypeRepo
from control.repository.postgres_objects import Statistic as StatisticRepo

from control.scheduler.fl_simple_scheduler import FLSimpleScheduler

from control.util.loader import Loader

import threading


class ScheduleManager:
    task_dispatcher: Dispatcher
    task_status = Task.WAITING

    def __init__(self, loader: Loader):

        self.loader = loader

        # load the Scheduler that will be used
        self.scheduler = FLSimpleScheduler(instance_types=self.loader.env)
        # self.__load_scheduler()

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

        # Keep Used EBS Volume
        self.ebs_volume_id = None

        # Vars Datetime to keep track of global execution time
        self.start_timestamp = None
        self.end_timestamp = None
        self.elapsed_time = None

        self.repo = PostgresRepo()

        # Semaphore
        self.semaphore = threading.Semaphore()
        self.semaphore_count = threading.Semaphore()

        # TRACKERS VALUES
        self.n_interruptions = 0
        self.n_sim_interruptions = 0

        self.timeout = False

        ''' ABORT FLAG'''
        self.abort = False

        self.server_task_dispatcher: Dispatcher
        self.client_tasks_dispatches: List[Dispatcher]
        self.terminated_dispatchers = []
        self.server_task_status = Task.WAITING
        self.client_tasks_status = []
        for i in range(self.loader.fl_server_task.n_clients):
            self.client_tasks_status.append(Task.WAITING)

        '''
                Build the initial dispatchers
                The class Dispatcher is responsible to manager the execution steps
                '''
        self.__build_dispatchers()

        # Prepare the control database and the folders structure in S3
        try:
            self.__prepare_execution()
        except Exception as e:
            logging.error(e)
            raise e

    # # PRE-EXECUTION FUNCTIONS

    # def __load_scheduler(self):
    #
    #     if self.loader.scheduler_name.upper() == Scheduler.CC:
    #         self.scheduler = CCScheduler(loader=self.loader)
    #
    #     elif self.loader.scheduler_name.upper() == Scheduler.IPDPS:
    #         self.scheduler = IPDPS(loader=self.loader)
    #
    #     if self.scheduler is None:
    #         logging.error("<Scheduler Manager {}_{}>: "
    #                       "ERROR - Scheduler {} not found".format(self.loader.job.job_id,
    #                                                               self.loader.execution_id,
    #                                                               self.loader.scheduler_name))
    #         Exception("<Scheduler Manager {}_{}>:  "
    #                   "ERROR - Scheduler {} not found".format(self.loader.job.job_id,
    #                                                           self.loader.execution_id,
    #                                                           self.loader.scheduler_name))

    def __build_dispatchers(self):

        instance_type, market = self.scheduler.get_server_initial_instance()

        # Create the Vm that will be used by the dispatcher
        vm = VirtualMachine(
            instance_type=instance_type,
            market=market,
            loader=self.loader
        )

        # than a dispatcher, that will execute the tasks, is create

        server_dispatcher = Dispatcher(vm=vm, loader=self.loader, type_task=Task.SERVER)

        # check if the VM need to be register on the simulator
        if self.loader.simulation_conf.with_simulation and vm.market == CloudManager.PREEMPTIBLE:
            self.simulator.register_vm(vm)

        self.server_task_dispatcher = server_dispatcher

        self.client_task_dispatchers = []

        for i in range(self.loader.fl_server_task.n_clients):
            instance_type, market = self.scheduler.get_client_initial_instance()

            # Create the Vm that will be used by the dispatcher
            vm = VirtualMachine(
                instance_type=instance_type,
                market=market,
                loader=self.loader
            )

            # than a dispatcher, that will execute the tasks, is create
            client_dispatcher = Dispatcher(vm=vm, loader=self.loader, type_task=Task.CLIENT)

            # check if the VM need to be register on the simulator
            if self.loader.simulation_conf.with_simulation and vm.market == CloudManager.PREEMPTIBLE:
                self.simulator.register_vm(vm)

            self.semaphore.acquire()

            self.client_task_dispatchers.append(client_dispatcher)

            self.semaphore.release()

    def __prepare_execution(self):
        """
                   Prepare control database and all directories to start the execution process
                """
        # get job from control database
        jobs_repo = self.repo.get_jobs(
            filter={
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

            assert len(tasks_repo) == len(
                self.loader.job.tasks), "Consistency error (number of tasks): {} <> {} ".format(
                len(tasks_repo), len(self.loader.job.tasks)
            )

            # check tasks consistency
            for t in tasks_repo:
                assert str(t.task_id) in self.loader.job.tasks, "Consistency error (task id): {}".format(
                    t.task_id)

                task = self.loader.job.tasks[str(t.task_id)]

                assert task.memory == t.memory, "Consistency error (task {} memory): {} <> {} ".format(
                    task.task_id, t.memory, task.memory)

                assert task.command == t.command, "Consistency error (task {} command): {} <> {} ".format(
                    task.task_id, t.command, task.command)

                assert task.io == t.io, "Consistency error (task {} io):".format(task.task_id, t.io, task.io)

        # Check Instances Type
        for key, instance_type in self.loader.env.items():

            types = self.repo.get_instance_type(filter={
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
        # """
        #    Prepare control database and all directories to start the execution process
        # """
        # # get server job from control database
        # tasks_repo = self.repo.get_tasks(
        #     filter={
        #         'task_id': self.loader.fl_server_task.task_id
        #     }
        # )
        #
        # # Check if server Job is already in the database
        # if len(tasks_repo) == 0:
        #     # add task to database
        #     self.__add_task_to_database(task_id=self.loader.fl_server_task.task_id,
        #                                 task_name=self.loader.fl_server_task.task_name,
        #                                 command=self.loader.fl_server_task.simple_command)
        # else:
        #     # Task is already in database
        #     # Check task and Instances consistency
        #     logging.info("<Scheduler Manager {}_{}>: - "
        #                  "Checking database consistency...".format(self.loader.fl_server_task.task_id,
        #                                                            self.loader.execution_id))
        #
        #     task_repo = tasks_repo[0]
        #
        #     assert task_repo.task_name == self.loader.fl_server_task.task_name, "Consistency error (task name): " \
        #                                                                        "{} <> {}"\
        #         .format(task_repo.task_name, self.loader.fl_server_task.task_name)
        #
        #     assert task_repo.command == self.loader.fl_server_task.simple_command, "Consistency error (task command): " \
        #                                                                           "{} <> {} "\
        #         .format(task_repo.command, self.loader.fl_server_task.simple_command)
        #
        # for i in range(self.loader.fl_server_task.n_clients):
        #     # get client job from control database
        #     tasks_repo = self.repo.get_tasks(
        #         filter={
        #             'task_id': self.loader.fl_client_tasks[i].task_id
        #         }
        #     )
        #
        #     # Check if Job is already in the database
        #     if len(tasks_repo) == 0:
        #         # add task to database
        #         self.__add_task_to_database(task_id=self.loader.fl_client_tasks[i].task_id,
        #                                     task_name=self.loader.fl_client_tasks[i].task_name,
        #                                     command=self.loader.fl_client_tasks[i].simple_command)
        #     else:
        #         # Task is already in database
        #         # Check task and Instances consistency
        #         logging.info("<Scheduler Manager {}_{}>: - "
        #                      "Checking database consistency...".format(self.loader.fl_client_tasks[i].task_id,
        #                                                                self.loader.execution_id))
        #
        #         task_repo = tasks_repo[0]
        #
        #         assert task_repo.task_name == self.loader.fl_client_tasks[i].task_name, "Consistency error (task name): " \
        #                                                                             "{} <> {}" \
        #             .format(task_repo.task_name, self.loader.fl_client_tasks[i].task_name)
        #
        #         assert task_repo.command == self.loader.fl_client_tasks[i].simple_command, "Consistency error (task command): " \
        #                                                                                "{} <> {} " \
        #             .format(task_repo.command, self.loader.fl_client_tasks[i].simple_command)
        #
        # # Check Instances Type
        # for key, instance_type in self.loader.env.items():
        #
        #     types = self.repo.get_instance_type(filter={
        #         'instance_type': key
        #     })
        #
        #     if len(types) == 0:
        #         # add instance to control database
        #         self.__add_instance_type_to_database(instance_type)
        #     else:
        #         # check instance type consistency
        #         inst_type_repo = types[0]
        #         assert inst_type_repo.vcpu == instance_type.vcpu, "Consistency error (vcpu instance {}): " \
        #                                                           "{} <> {} ".format(key,
        #                                                                              inst_type_repo.vcpu,
        #                                                                              instance_type.vcpu)
        #
        #         assert inst_type_repo.memory == instance_type.memory, "Consistency error (memory instance {}):" \
        #                                                               "{} <> {}".format(key,
        #                                                                                 inst_type_repo.memory,
        #                                                                                 instance_type.memory)

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
                    command=task.command,
                )
            )

    def __add_task_to_database(self, task_id, task_name, command):
        """Record a Task to the controlgpu database"""

        task_repo = TaskRepo(
            task_id=task_id,
            task_name=task_name,
            command=command
        )

        self.repo.add_task(task_repo)

    def __add_instance_type_to_database(self, instance_type):
        self.repo.add_instance_type(
            InstanceTypeRepo(
                type=instance_type.type,
                provider=instance_type.provider
            )
        )

    '''
    HANDLES FUNCTIONS
    '''

    def __interruption_handle(self):
        pass
        # # Move task to other VM
        # # self.semaphore.acquire()
        #
        # if not self.loader.cudalign_task.has_task_finished():
        #     self.loader.cudalign_task.stop_execution()
        #
        # # logging.info("Entrou no interruption_handle")
        #
        # # getting volume-id
        # if self.loader.file_system_conf.type == EC2Manager.EBS:
        #     self.ebs_volume_id = self.task_dispatcher.vm.volume_id
        #
        # # logging.info("Pegou o id do EBS: {}".format(self.ebs_volume_id))
        #
        # # See in which VM we wiil restart
        # current_time = self.start_timestamp - datetime.now()
        #
        # instance_type, market = self.scheduler.choose_restart_best_instance_type(
        #     cudalign_task=self.loader.cudalign_task,
        #     deadline=self.loader.deadline_seconds,
        #     current_time=current_time.total_seconds()
        # )
        #
        # # logging.info("Escolheu instancia {} do tipo {}".format(instance_type.type, market))
        #
        # if self.loader.cudalign_task.has_task_finished():
        #     new_vm = VirtualMachine(
        #         instance_type=instance_type,
        #         market=market,
        #         loader=self.loader,
        #         volume_id=self.ebs_volume_id
        #     )
        #
        #     # logging.info("Criou a nova vm!")
        #
        #     dispatcher = Dispatcher(vm=new_vm, loader=self.loader)
        #
        #     # check if the VM need to be register on the simulator
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

    def __terminated_handle(self):
        pass
        # # Move task to others VM
        # # self.semaphore.acquire()
        #
        # if not self.loader.cudalign_task.has_task_finished():
        #     self.loader.cudalign_task.stop_execution()
        #
        # # logging.info("Entrou no terminated_handle")
        #
        # # getting volume-id
        # if self.loader.file_system_conf.type == EC2Manager.EBS:
        #     self.ebs_volume_id = self.task_dispatcher.vm.volume_id
        #
        # # logging.info("Pegou o id do EBS: {}".format(self.ebs_volume_id))
        #
        # # See in which VM will restart
        # current_time = self.start_timestamp - datetime.now()
        #
        # instance_type, market = self.scheduler.choose_restart_best_instance_type(
        #     cudalign_task=self.loader.cudalign_task,
        #     deadline=self.loader.deadline_seconds,
        #     current_time=current_time.total_seconds()
        # )
        #
        # # logging.info("Escolheu instancia {} do tipo {}".format(instance_type.type, market))
        #
        # if not self.loader.cudalign_task.has_task_finished():
        #     new_vm = VirtualMachine(
        #         instance_type=instance_type,
        #         market=market,
        #         loader=self.loader,
        #         volume_id=self.ebs_volume_id
        #     )
        #
        #     # logging.info("Criou a nova vm!")
        #
        #     dispatcher = Dispatcher(vm=new_vm, loader=self.loader)
        #
        #     # check if the VM need to be register on the simulator
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

        logging.info("<Scheduler Manager {}_{}>: - EVENT_HANDLE "
                     "Instance: '{}', Type: '{}', Market: '{}',"
                     "Event: '{}'".format(self.loader.fl_server_task.task_id,
                                          self.loader.execution_id,
                                          self.task_dispatcher.vm.instance_id,
                                          self.task_dispatcher.vm.type,
                                          self.task_dispatcher.vm.market,
                                          event.value))

        if event.value == CloudManager.IDLE:
            logging.info("<Scheduler Manager {}_{}>: - Calling Idle Handle".format(self.loader.fl_server_task.task_id,
                                                                                   self.loader.execution_id))

            self.loader.fl_server_task.finish_execution()
            self.server_task_status = Task.FINISHED
            for i in range(self.loader.fl_server_task.n_clients):
                self.loader.fl_client_tasks[i].finish_execution()
                self.client_tasks_status[i] = Task.FINISHED
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
        # elif event.value in [CloudManager.TERMINATED, CloudManager.ERROR]:
        #     logging.info("<Scheduler Manager {}_{}>: - Calling Terminate Handle"
        #                  .format(self.loader.cudalign_task.task_id, self.loader.execution_id))
        #     if not self.task_dispatcher.vm.marked_to_interrupt:
        #         self.n_sim_interruptions += 1
        #     self.__terminated_handle()

        elif event.value in CloudManager.ABORT:
            self.abort = True

    '''
    CHECKERS FUNCTIONS
    '''

    def __checkers(self):
        # Checker loop
        # Checker if all dispatchers have finished the execution
        while self.task_status != Task.FINISHED:

            if self.abort:
                break

            time.sleep(5)

    '''
    Manager Functions
    '''

    def __start_dispatcher(self):
        self.semaphore.acquire()

        # Starting working dispatcher
        self.server_task_dispatcher.main_thread.start()
        for i in range(self.loader.fl_server_task.n_clients):
            self.client_task_dispatchers[i].main_thread.start()
        # self.task_dispatcher.waiting_work.set()

        self.semaphore.release()

    def __terminate_dispatcher(self):

        if self.loader.debug_conf.debug_mode:
            logging.warning(100 * "#")
            logging.warning("\t<DEBUG MODE>: WAITING COMMAND TO TERMINATE -  PRESS ENTER")
            logging.warning(100 * "#")

            input("")

        logging.info("")
        logging.info("<Scheduler Manager {}_{}>: - Start termination process... "
                     .format(self.loader.fl_server_task.task_id, self.loader.execution_id))

        # terminate simulation
        if self.loader.simulation_conf.with_simulation:
            self.simulator.stop_simulation()

        # self.semaphore.acquire()

        # Terminate DISPATCHER
        logging.info("<Scheduler Manager {}_{}>: - "
                     "Terminating Dispatcher".format(self.loader.fl_server_task.task_id,
                                                     self.loader.execution_id))

        self.task_dispatcher.debug_wait_command = False

        self.task_dispatcher.working = False
        # self.task_dispatcher.waiting_work.set()

        # Confirm Termination
        logging.info("<Scheduler Manager {}_{}>: - Waiting Termination process..."
                     .format(self.loader.cudalign_task.task_id, self.loader.execution_id))

        self.task_dispatcher.debug_wait_command = False
        # waiting thread to terminate

        self.task_dispatcher.main_thread.join()

        # getting volume-id
        if self.loader.file_system_conf.type == EC2Manager.EBS:
            self.ebs_volume_id = self.task_dispatcher.vm.volume_id

        self.terminated_dispatchers.append(self.task_dispatcher)

        # self.semaphore.release()

    def __end_of_execution(self):

        # end of execution
        self.end_timestamp = datetime.now()
        self.elapsed_time = (self.end_timestamp - self.start_timestamp)

        logging.info("<Scheduler Manager {}_{}>: - Waiting Termination...".format(self.loader.cudalign_task.task_id,
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
            execution_info = "    Task: {} Execution: {} Scheduler: SimpleScheduler    "\
                .format(self.loader.cudalign_task.task_id, self.loader.execution_id)
        else:
            execution_info = "    Job: {} Execution: {} Scheduler: SimpleScheduler" \
                             " - EXECUTION ABORTED    ".format(self.loader.cudalign_task.task_id,
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
            logging.warning("The following EBS VOLUMES will note be deleted by HADS: ")
            logging.warning("\t-> {}".format(self.ebs_volume_id))

        logging.info("")
        logging.info(len(execution_info) * "#")

        status = 'success'

        if self.abort:
            status = 'aborted'

        self.repo.add_statistic(
            StatisticRepo(execution_id=self.loader.execution_id,
                          task_id=self.loader.cudalign_task.task_id,
                          status=status,
                          start=self.start_timestamp,
                          end=self.end_timestamp,
                          deadline=self.loader.deadline_timedelta,
                          cost=cost)
        )

        self.repo.close_session()

        if self.abort:
            error_msg = "<Scheduler Manager {}_{}>: - " \
                        "Check all log-files. Execution Aborted".format(self.loader.cudalign_task.task_id,
                                                                        self.loader.execution_id)
            logging.error(error_msg)
            raise Exception

    def start_execution(self):
        # subscriber events_handle
        subscribers.append(self.__event_handle)

        self.start_timestamp = datetime.now()
        # UPDATE DATETIME DEADLINE

        logging.info("<Scheduler Manager {}_{}>: - Starting Execution.".format(self.loader.fl_server_task.task_id,
                                                                               self.loader.execution_id))
        logging.info("")

        self.__start_dispatcher()

        # Call checkers loop
        self.__checkers()

        self.__terminate_dispatcher()

        self.__end_of_execution()
