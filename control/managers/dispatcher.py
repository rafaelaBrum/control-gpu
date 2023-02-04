from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

# import tarfile

from control.domain.task import Task
from control.domain.job import Job

# from control.scheduler.queue import Queue

# from control.util.linked_list import LinkedList
from control.util.event import Event
from control.util.loader import Loader

from control.daemon.daemon_manager import Daemon
from control.daemon.communicator import Communicator

from control.repository.postgres_repo import PostgresRepo
from control.repository.postgres_objects import Execution as ExecutionRepo
from control.repository.postgres_objects import Instance as InstanceRepo
# from control.repository.postgres_objects import InstanceStatus as InstanceStatusRepo
# from control.repository.postgres_objects import InstanceStatistic as InstanceStatisticRepo
# from control.repository.postgres_objects import TaskStatistic as TaskStatisticRepo

# from typing import List
import threading
import time
import logging
# import os
# import math

from datetime import datetime

from zope.event import notify


class Executor:

    def __init__(self, task: Task, vm: VirtualMachine, loader: Loader, type_task):

        self.loader = loader

        self.task = task
        self.vm = vm

        self.type_task = type_task

        self.repo = None
        # Execution Status
        self.status = Task.WAITING

        # socket.communicator
        # used to send commands to the ec2 instance
        self.communicator = Communicator(host=self.vm.instance_public_ip,
                                         port=self.loader.communication_conf.socket_port)

        """Track INFO """
        # used to abort the execution loop
        self.stop_signal = False
        # checkpoint tracker
        self.next_checkpoint_datetime = None

        self.thread = threading.Thread(target=self.__run, daemon=True)
        self.thread_executing = False

    def update_status_table(self):
        """
        Update Execution Table
        Call if task status change
        """
        # Update Execution Status Table
        self.repo.add_execution(
            ExecutionRepo(
                execution_id=self.loader.execution_id,
                job_id=self.loader.job.job_id,
                task_id=self.task.task_id,
                instance_id=f'{self.vm.instance_id}_{self.loader.execution_id}',
                timestamp=datetime.now(),
                status=self.status
            )
        )

        # repo.close_session()

    def __run(self):
        # START task execution

        # logging.info("<Executor {}-{}>: __run function".format(self.task.task_id, self.vm.instance_id))

        self.repo = PostgresRepo()
        # current_time = None
        action = Daemon.START

        # if self.task.has_checkpoint:
        #     action = Daemon.RESTART
        try:
            # logging.info("<Executor {}-{}>: Sending action Daemon.START".format(self.task.task_id,
            #                                                                     self.vm.instance_id))
            # if self.type_task == Job.CLIENT:
            #     logging.info("<Executor {}-{}>: dict_info {}".format(self.task.task_id,
            #                                                          self.vm.instance_id,
            #                                                          self.dict_info))
            self.communicator.send(action=action, value=self.dict_info)
            current_time = datetime.now()
            # logging.info("<Executor {}-{}>: Action Daemon.START sent".format(self.task.task_id, self.vm.instance_id))
        except Exception as e:
            logging.error(e)
            self.__stopped(Task.ERROR)
            return

        # if task was started with success
        # start execution loop
        if self.communicator.response['status'] == 'success':

            self.status = Task.EXECUTING

            # if action == Daemon.START:
            #     self.status = Task.EXECUTING
            # else:
            #     self.status = Task.RESTARTED

            # self.update_status_table()

            # self.stop_signal = True

            logging.info("<Executor {}-{}>: Begin execution loop".format(self.task.task_id, self.vm.instance_id))

            try:

                # start task execution Loop
                while self.status in (Task.EXECUTING, Task.RESTART) and not self.stop_signal \
                        and self.vm.state == CloudManager.RUNNING:

                    try:
                        # logging.info(
                        #     "<Executor {}-{}>: Trying to get task status".format(self.task.task_id,
                        #                                                          self.vm.instance_id))
                        command_status, current_stage = self.__get_task_status()
                        # logging.info(
                        #     "<Executor {}-{}>: Command status {}".format(self.task.task_id, self.vm.instance_id,
                        #                                                  command_status))

                        instance_action = None
                        if self.vm.market == CloudManager.PREEMPTIBLE:
                            # logging.info(
                            #     "<Executor {}-{}>: Trying to get instance action".format(self.task.task_id,
                            #                                                              self.vm.instance_id))
                            instance_action = self.__get_instance_action()
                            # logging.info(
                            #     "<Executor {}-{}>: Instance action {}".format(self.task.task_id, self.vm.instance_id,
                            #                                                   instance_action))

                        # if self.loader.checkpoint_conf.with_checkpoint \
                        #     and self.vm.market == CloudManager.PREEMPTIBLE and self.task.do_checkpoint:
                        #     self.__checkpoint_task()

                    except Exception as e:
                        logging.error(e)
                        self.__stopped(Task.ERROR)
                        return

                    # check task status
                    if command_status is not None and command_status == 'finished':

                        self.status = status = Task.FINISHED

                        self.task.finish_execution()
                        self.__stopped(status)
                        return

                    if command_status is not None and command_status == 'running':
                        elapsed_time = datetime.now() - current_time
                        current_time = current_time + elapsed_time
                        self.task.update_execution_time(elapsed_time.total_seconds())

                    if ((self.vm.instance_type.provider == CloudManager.EC2 and
                         instance_action is not None and
                         instance_action != 'none') or
                            (self.vm.instance_type.provider == CloudManager.GCLOUD and
                             instance_action == 'TRUE')):
                        self.vm.interrupt()
                        self.__stopped(Task.INTERRUPTED)
                        return

                    if command_status is not None and command_status != 'running':
                        task_status = False
                        if self.type_task == Job.CLIENT:
                            task_status = self.__restart_client_task()
                        if self.type_task == Job.SERVER or not task_status:
                            self.task.stop_execution()
                            self.__stopped(Task.RUNTIME_ERROR)
                            return

                    time.sleep(10)
            except Exception as e:
                logging.error(e)
                self.__stopped(Task.RUNTIME_ERROR)
                return

            if self.status != Task.FINISHED:
                self.task.stop_execution()

        # if kill signal than checkpoint task (SIMULATION)
        # if self.stop_signal:
        #     # check is task is running
        #     try:
        #         command_status, current_stage = self.__get_task_status()
        #         if command_status is not None and command_status == 'running':
        #             self.__stop()  # Checkpoint and stop task
        #             # self.__stopped(Task.HIBERNATED)
        #         # else:
        #             # self.__stopped(Task.FINISHED)
        #
        #     except Exception as e:
        #         logging.error(e)
        #         # self.__stopped(Task.STOP_SIGNAL)
        #
        #     return

        # self.__stopped(Task.ERROR)

    def __stop(self):
        # START task execution

        self.repo = PostgresRepo()

        action = Daemon.STOP

        try:
            self.communicator.send(action=action, value=self.dict_info)
        except Exception as e:
            logging.error(e)
            self.__stopped(Task.ERROR)
            return

    def __stopped(self, status):
        self.status = status
    #     # update execution time
    #
    #     # if task had Migrated, not to do
    #     if self.status == Task.MIGRATED:
    #         self.repo.close_session()
    #         return
    #
        self.status = status

        self.update_status_table()
        # close repo
        self.repo.close_session()

        # Check if condition is true to checkpoint the task

    # def __checkpoint_task(self):
    #
    #     if self.next_checkpoint_datetime is None:
    #         # compute next_checkpoint datetime
    #         self.next_checkpoint_datetime = datetime.now() + timedelta(seconds=self.task.checkpoint_interval)
    #
    #     elif datetime.now() > self.next_checkpoint_datetime:
    #
    #         self.__checkpoint()
    #         self.next_checkpoint_datetime = datetime.now() + timedelta(seconds=self.task.checkpoint_interval)

    # def __checkpoint(self, stop_task=False):
    #
    #     for i in range(3):
    #         try:
    #
    #             action = Daemon.CHECKPOINT_STOP if stop_task else Daemon.CHECKPOINT
    #
    #             logging.info("<Executor {}-{}>: Checkpointing task...".format(self.task.task_id,
    #                                                                           self.vm.instance_id))
    #
    #             start_ckp = datetime.now()
    #             self.communicator.send(action, value=self.dict_info)
    #
    #             if self.communicator.response['status'] == 'success':
    #                 end_ckp = datetime.now()
    #
    #                 logging.info("<Executor {}-{}>: Checkpoint with success. Time: {}".format(self.task.task_id,
    #                                                                                           self.vm.instance_id,
    #                                                                                           end_ckp - start_ckp))
    #                 self.task.has_checkpoint = True
    #                 self.task.update_task_time()
    #
    #             return
    #         except:
    #             pass
    #
    #     raise Exception("<Executor {}-{}>: Checkpoint error".format(self.task.task_id, self.vm.instance_id))

    def __get_task_status(self):

        for i in range(3):

            try:

                self.communicator.send(action=Daemon.STATUS,
                                       value=self.dict_info)

                result = self.communicator.response

                command_status = result['value']['status']
                current_stage = result['value']['current_stage']

                return command_status, current_stage
            except Exception:
                logging.error("<Executor {}-{}>: Get task Status {}/3".format(self.task.task_id,
                                                                              self.vm.instance_id,
                                                                              i + 1))
                time.sleep(1)

        raise Exception("<Executor {}-{}>: Get task status error".format(self.task.task_id, self.vm.instance_id))

    def __get_instance_action(self):

        for i in range(3):

            try:

                self.communicator.send(action=Daemon.INSTANCE_ACTION,
                                       value=self.dict_info)

                result = self.communicator.response

                instance_action = result['value']

                return instance_action
            except Exception:
                logging.error("<Executor {}-{}>: Get instance action {}/3".format(self.task.task_id,
                                                                                  self.vm.instance_id,
                                                                                  i + 1))
                time.sleep(1)

        raise Exception("<Executor {}-{}>: Get instance action error".format(self.task.task_id, self.vm.instance_id))

    def __restart_client_task(self):
        self.status = Task.RESTART
        for i in range(20):
            try:
                # logging.info("<Executor {}-{}>: restarting task".format(self.task.task_id,
                                                                        # self.vm.instance_id))
                # if self.type_task == Job.CLIENT:
                #     logging.info("<Executor {}-{}>: dict_info {}".format(self.task.task_id,
                #                                                          self.vm.instance_id,
                #                                                          self.dict_info))
                self.communicator.send(action=Daemon.START, value=self.dict_info)
            except Exception as e:
                logging.error(e)
            # if task was started with success
            # start execution loop
            if self.communicator.response['status'] == 'success':
                # logging.info("<Executor {}-{}>: Successfully restarted task".format(self.task.task_id,
                #                                                                     self.vm.instance_id))
                self.status = Task.EXECUTING
                return True
            time.sleep(300)
        return False

    # def __get_task_usage(self):
    #     for i in range(3):
    #         try:
    #             self.communicator.send(action=Daemon.TASK_USAGE,
    #                                    value=self.dict_info)
    #
    #             result = self.communicator.response
    #
    #             usage = None
    #
    #             if result['status'] == 'success':
    #                 usage = result['value']
    #
    #             return usage
    #         except:
    #             logging.error(
    #                 "<Executor {}-{}>: Get task Usage {}/3".format(self.task.task_id, self.vm.instance_id, i + 1))
    #             time.sleep(1)
    #
    #     raise Exception("<Executor {}-{}>: Get task usage error".format(self.task.task_id, self.vm.instance_id))

    # def __to_megabyte(self, str):
    #
    #     pos = str.find('MiB')
    #
    #     if pos == -1:
    #         pos = str.find('GiB')
    #     if pos == -1:
    #         pos = str.find('KiB')
    #     if pos == -1:
    #         pos = str.find('B')
    #
    #     memory = float(str[:pos])
    #     index = str[pos:]
    #
    #     to_megabyte = {
    #         "GiB": 1073.742,
    #         "MiB": 1.049,
    #         "B": 1e+6,
    #         "KiB": 976.562
    #     }
    #
    #     return to_megabyte[index] * memory

    @property
    def dict_info(self):

        info = {
            "task_id": self.task.task_id,
            "command": self.task.command,
            "server_ip": self.task.server_ip,
            "cpu": self.vm.instance_type.vcpu,
            "gpu": self.vm.instance_type.count_gpu
        }

        return info


class Dispatcher:

    def __init__(self, vm: VirtualMachine, loader: Loader, type_task, client_id, needs_external_transfer=False):
        self.loader = loader

        self.vm: VirtualMachine = vm  # Class that control a Virtual machine on the cloud
        # self.queue = queue  # Class with the scheduling plan

        self.type_task = type_task
        self.client_id = client_id
        self.client = None
        if client_id < self.loader.job.num_clients:
            self.client = self.loader.job.client_tasks[client_id]

        # Control Flags
        self.working = False
        # flag to indicate that the instance is ready to execute
        self.ready = False
        # indicate that VM hibernates
        self.interrupted = False

        # debug flag indicates that the dispatcher should wait for the shutdown command
        self.debug_wait_command = self.loader.debug_conf.debug_mode

        # migration count
        self.migration_count = 0

        # needs_external_transfer (send ckpt to restart task)
        self.needs_external_transfer = needs_external_transfer
        self.start_execution = not self.needs_external_transfer

        '''
        List that determine the execution order of the
        tasks that will be executed in that dispatcher
        '''

        # threading event to waiting for tasks to execute
        # self.waiting_work = threading.Event()
        self.semaphore = threading.Semaphore()

        self.main_thread = threading.Thread(target=self.__execution_loop, daemon=True)

        self.repo = PostgresRepo()
        self.least_status = None
        self.timestamp_status_update = None

        self.executor: Executor = None

        # self.stop_period = None

    def update_server_ip(self, server_ip):
        if self.type_task == Job.CLIENT:
            self.loader.job.client_tasks[self.client_id].server_ip = server_ip
            if self.executor is not None:
                self.executor.task.server_ip = server_ip
            # if self.main_thread.is_alive():
            #     logging.error("<Dispatcher {}>: Thread still alive".format(self.vm.instance_id))
            # self.main_thread = threading.Thread(target=self.__restart_client_execution_loop, daemon=True)

    # def __get_instance_usage(self):
    #     memory = 0
    #     cpu = 0
    #
    #     communicator = Communicator(self.vm.instance_public_ip,
    #                                 self.loader.communication_conf.socket_port)
    #
    #     info = {
    #         "task_id": 0,
    #         "command": '',
    #         'cpu_quota': 0
    #     }
    #
    #     max_attempt = 1
    #
    #     for i in range(max_attempt):
    #         try:
    #             communicator.send(action=Daemon.INSTANCE_USAGE, value=info)
    #
    #             result = communicator.response
    #
    #             if result['status'] == 'success':
    #                 memory = float(result['value']['memory'])
    #                 cpu = float(result['value']['cpu'])
    #         except:
    #             logging.error("<Dispatcher {}>: Get Instance Usage {}/{}".format(self.vm.instance_id,
    #                                                                              i + 1,
    #                                                                              max_attempt))
    #
    #     return cpu, memory

    # def __update_instance_status_table(self, state=None):
    #     """
    #     Update Instance Status table
    #     """
    #     if state is None:
    #         state = self.vm.state
    #
    #     # Check if the update have to be done due to the time
    #
    #     time_diff = None
    #     if self.timestamp_status_update is not None:
    #         time_diff = datetime.now() - self.timestamp_status_update
    #
    #     if self.least_status is None or self.least_status != state or \
    #         time_diff > timedelta(seconds=self.loader.scheduler_conf.status_update_time):
    #         cpu = 0.0
    #         memory = 0.0
    #         # cpu, memory = self.__get_instance_usage()
    #         # Update Instance_status Table
    #         self.repo.add_instance_status(InstanceStatusRepo(instance_id=self.vm.instance_id,
    #                                                          timestamp=datetime.now(),
    #                                                          status=state,
    #                                                          memory_footprint=memory,
    #                                                          cpu_usage=cpu,
    #                                                          cpu_credit=self.vm.get_cpu_credits()))
    #
    #         self.timestamp_status_update = datetime.now()
    #         self.least_status = self.vm.state

    # def __update_instance_statistics_table(self):
    #
    #     self.repo.add_instance_status(InstanceStatisticRepo(instance_id=self.vm.instance_id,
    #                                                         deploy_overhead=self.vm.deploy_overhead.seconds,
    #                                                         termination_overhead=self.vm.terminate_overhead.seconds,
    #                                                         uptime=self.vm.uptime.seconds))

    def __notify(self, value):

        kwargs = {'instance_id': self.vm.instance_id,
                  'dispatcher': self,
                  'type_task': self.type_task,
                  'client_id': self.client_id}

        notify(
            Event(
                event_type=Event.INSTANCE_EVENT,
                value=value,
                **kwargs
            )
        )

    def __prepare_daemon(self):
        attempt = 1
        while True:
            time.sleep(self.loader.communication_conf.retry_interval)

            try:
                communicator = Communicator(host=self.vm.instance_public_ip,
                                            port=self.loader.communication_conf.socket_port)
                communicator.send(action=Daemon.TEST, value={'task_id': None, 'command': None,
                                                             'server_ip': None, 'cpu': None, 'gpu': None})

                if communicator.response['status'] == 'success':
                    return True

            except Exception as e:
                if attempt > self.loader.communication_conf.repeat:
                    logging.error(e)
                    return False

            if attempt <= self.loader.communication_conf.repeat:
                logging.info('<Dispatcher {}>: Trying Daemon handshake... attempt {}/{}'
                             ''.format(self.vm.instance_id, attempt,
                                       self.loader.communication_conf.repeat))
            else:
                logging.info('<Dispatcher {}>: Daemon handshake MAX ATTEMPT ERROR'
                             ''.format(self.vm.instance_id, attempt,
                                       self.loader.communication_conf.repeat))

            attempt += 1

    def __execution_loop(self):

        # Start the VM in the cloud
        logging.info("Deploying VM of {} in {}".format(self.type_task, self.vm.zone))
        if self.type_task == Job.SERVER:
            logging.info("Vms region {} and image_id {}".format(self.vm.region.region, self.vm.region.server_image_id))
        else:
            logging.info("Vms region {} and image_id {}".format(self.vm.region.region, self.vm.region.client_image_id))

        if self.client is None:
            status = self.vm.deploy(type_task=self.type_task)
        else:
            status = self.vm.deploy(type_task=self.type_task, dataset_urn=self.client.dataset_urn)

        if not status:
            for zone in self.vm.region.zones:
                self.vm.instance_id = None
                self.vm.zone = zone
                self.vm.instance_type.zone = zone
                logging.info("Deploying VM of {} in {}".format(self.type_task, self.vm.zone))
                if self.type_task == Job.SERVER:
                    logging.info(
                        "Vms region {} and image_id {}".format(self.vm.region.region, self.vm.region.server_image_id))
                else:
                    logging.info(
                        "Vms region {} and image_id {}".format(self.vm.region.region, self.vm.region.client_image_id))
                status = self.vm.deploy(type_task=self.type_task)
                if status:
                    break

        # self.expected_makespan_timestamp = self.vm.start_time + timedelta(seconds=self.queue.makespan_seconds)

        # update instance_repo
        self.repo.add_instance(InstanceRepo(id=f'{self.vm.instance_id}_{self.loader.execution_id}',
                                            type=self.vm.instance_type.type,
                                            region=self.vm.instance_type.region,
                                            zone=self.vm.instance_type.zone,
                                            market=self.vm.market,
                                            ebs_volume=self.vm.volume_id,
                                            price=self.vm.price))

        # self.__update_instance_status_table()

        if status:

            self.working = True

            try:
                self.vm.prepare_vm(self.type_task, self.client)
                status = self.__prepare_daemon()
            except Exception as e:
                logging.error(e)

                # stop working process
                # self.waiting_work.clear()
                # Notify abort!
                self.__notify(CloudManager.ABORT)

            if not status:
                logging.error("<Dispatcher {}>: Daemon not working!".format(self.vm.instance_id))
                self.__notify(CloudManager.ABORT)
                self.working = False

            # indicate that the VM is ready to execute
            self.vm.ready = self.ready = True
            if self.type_task == Job.SERVER:
                task = self.loader.job.server_task
            elif self.type_task == Job.CLIENT:
                task = self.loader.job.client_tasks[self.client_id]
            else:
                task = None

            # logging.info('<Dispatcher {}>: needs_external_transfer? {}'.format(self.vm.instance_id,
            #                                                                    self.needs_external_transfer))

            if self.needs_external_transfer:
                while not self.start_execution:
                    continue

            # logging.info('<Dispatcher {}>: starting execution'.format(self.vm.instance_id))
            #
            # logging.info('<Dispatcher {}>: task finished? {}'.format(self.vm.instance_id, task.has_task_finished()))

            if not task.has_task_finished() and self.working:
                if self.vm.state == CloudManager.RUNNING:

                    self.semaphore.acquire()
                    # # check running tasks
                    # self.__update_running_executors()

                    # logging.info('<Dispatcher {}>: task is running? {}'.format(self.vm.instance_id,
                    #                                                            task.is_running()))

                    if not task.is_running():
                        self.executor = Executor(
                            task=task,
                            vm=self.vm,
                            loader=self.loader,
                            type_task=self.type_task
                        )
                        # start the executor loop to execute the task
                        self.executor.thread.start()
                        task.start_execution(self.vm.instance_type.type)

                    # logging.info('<Dispatcher {}>: task is running? {}'.format(self.vm.instance_id,
                    #                                                            task.is_running()))

                    self.semaphore.release()

            while self.working and not task.has_task_finished() and task.is_running():
                # waiting for work
                # self.waiting_work.wait()
                #
                # self.waiting_work.clear()
                if not self.working:
                    break

                # # execution loop
                # self.semaphore.acquire()
                # self.semaphore.release()

                # Error: instance was not deployed or was terminated
                if self.vm.state in (CloudManager.ERROR, CloudManager.SHUTTING_DOWN,
                                     CloudManager.TERMINATED, CloudManager.STOPPING):
                    # waiting running tasks
                    self.executor.thread.join()
                    # VM was not created, raise a event
                    self.__notify(CloudManager.TERMINATED)

                    break

                # elif self.vm.state == CloudManager.STOPPING:
                #     # waiting running tasks
                #     self.executor.thread.join()
                #
                #     self.resume = False
                #
                #     self.__notify(CloudManager.STOPPING)

                elif self.vm.state == CloudManager.STOPPED:
                    # STOP and CHECKPOINT all tasks
                    self.executor.stop_signal = True

                    # waiting running tasks
                    self.executor.thread.join()

                    self.resume = False

                    self.__notify(CloudManager.STOPPED)

                    # break

            if self.vm.state == CloudManager.RUNNING:

                # self.__update_instance_status_table(state=CloudManager.IDLE)
                self.__notify(CloudManager.IDLE)

                while self.debug_wait_command:
                    time.sleep(5)

                # self.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

            self.repo.close_session()

        else:
            # Error to start VM
            logging.error("<Dispatcher> Instance type: {} Was not started".format(self.vm.instance_type.type))
            self.vm.failed_to_created = True
            self.__notify(CloudManager.ERROR)

    def __restart_client_execution_loop(self):

        self.working = True

        status = False

        try:
            status = self.__prepare_daemon()
        except Exception as e:
            logging.error(e)

            # stop working process
            # self.waiting_work.clear()
            # Notify abort!
            self.__notify(CloudManager.ABORT)

        if not status:
            logging.error("<Dispatcher {}>: Daemon not working!".format(self.vm.instance_id))
            self.__notify(CloudManager.ABORT)
            self.working = False

        # indicate that the VM is ready to execute
        self.vm.ready = self.ready = True
        if self.type_task == Job.CLIENT:
            task = self.loader.job.client_tasks[self.client_id]
        else:
            task = None

        if not task.has_task_finished() and self.working:
            if self.vm.state == CloudManager.RUNNING:

                self.semaphore.acquire()
                # # check running tasks
                # self.__update_running_executors()

                if not task.is_running():
                    self.executor = Executor(
                        task=task,
                        vm=self.vm,
                        loader=self.loader,
                        type_task=self.type_task
                    )
                    # start the executor loop to execute the task
                    self.executor.thread.start()
                    task.start_execution(self.vm.instance_type.type)

                self.semaphore.release()

        while self.working and not task.has_task_finished() and task.is_running():
            # waiting for work
            # self.waiting_work.wait()
            #
            # self.waiting_work.clear()
            if not self.working:
                break

            # # execution loop
            # self.semaphore.acquire()
            # self.semaphore.release()

            # Error: instance was not deployed or was terminated
            if self.vm.state in (CloudManager.ERROR, CloudManager.SHUTTING_DOWN,
                                 CloudManager.TERMINATED, CloudManager.STOPPING):
                # waiting running tasks
                self.executor.thread.join()
                # VM was not created, raise a event
                self.__notify(CloudManager.TERMINATED)

                break

            # elif self.vm.state == CloudManager.STOPPING:
            #     # waiting running tasks
            #     self.executor.thread.join()
            #
            #     self.resume = False
            #
            #     self.__notify(CloudManager.STOPPING)

            elif self.vm.state == CloudManager.STOPPED:
                # STOP and CHECKPOINT all tasks
                self.executor.stop_signal = True

                # waiting running tasks
                self.executor.thread.join()

                self.resume = False

                self.__notify(CloudManager.STOPPED)

                # break

        if self.vm.state == CloudManager.RUNNING:

            # self.__update_instance_status_table(state=CloudManager.IDLE)
            self.__notify(CloudManager.IDLE)

            while self.debug_wait_command:
                time.sleep(5)

            # self.vm.terminate(delete_volume=self.loader.file_system_conf.ebs_delete)

        self.repo.close_session()

    def update_rounds(self, rounds_done):
        if self.type_task == Job.SERVER:
            self.loader.job.server_task.n_rounds = self.loader.job.server_task.n_rounds - rounds_done
            logging.info('<Dispatcher {}>: Updating num rounds to {}'.format(self.vm.instance_id,
                                                                             self.loader.job.server_task.n_rounds))
            aux_split = self.loader.job.server_task.command.split(' --rounds ')
            space_split = aux_split[-1].split(" ")
            aux_split[-1] = " ".join(space_split[1:])
            final_command = aux_split[0] + f" --rounds {self.loader.job.server_task.n_rounds} " + aux_split[-1]
            print("final_command", final_command)
            self.loader.job.server_task.command = final_command
            if self.executor is not None:
                self.executor.task.command = final_command
