#!/usr/bin/env python3
from datetime import datetime
from datetime import timedelta

# from pathlib import Path

import cherrypy

import argparse
import subprocess
# import re

# import shutil
import os
import logging


class Daemon:
    # CHECKPOINT_LIMIT = 3

    START = 'start'
    STATUS = 'status'
    STOP = 'stop'
    # TASK_USAGE = 'task_usage'
    # INSTANCE_USAGE = 'instance_usage'
    TEST = 'test'
    SUCCESS = 'success'
    ERROR = 'error'

    def __init__(self, vm_user, root_path, work_path, task_id, execution_id, instance_id):
        self.vm_user = vm_user

        self.work_path = work_path

        self.task_id = task_id
        self.execution_id = execution_id
        self.instance_id = instance_id

        self.root_path = os.path.join(root_path, "{}_{}".format(self.task_id, self.execution_id))

        self.__prepare_logging()

    # waiting for commands

    def __prepare_logging(self):

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel('INFO')

        file_name = os.path.join(self.root_path,
                                 "{}_{}_{}.log".format(self.task_id, self.execution_id, self.instance_id))

        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    def handle_command(self, action, value):

        task_id = value['task_id']
        command = value['command']
        session = ''

        if command is not None:
            session = command.split()[0]

        session_name = "Session_{}_{}_{}_{}".format(
            session,
            self.task_id,
            self.execution_id,
            task_id
        )

        vm_name = "VM_{}_{}_{}".format(
            self.task_id,
            self.execution_id,
            task_id
        )
        # task_path = os.path.join(self.root_path, "{}/".format(task_id))
        # data_path = os.path.join(self.root_path, "{}/data/".format(task_id))
        # backup_path = os.path.join(self.root_path, "{}/backup/".format(task_id))

        logging.info("VM {}: Action {}".format(vm_name, action))

        start_time = datetime.now()

        if action == Daemon.START:

            # Starting job
            try:

                self.__start_command(session_name, command)

                status_return = Daemon.SUCCESS
                value_return = "VM '{}' starts task with success".format(vm_name)

            except Exception as e:
                logging.error(e)
                status_return = Daemon.ERROR
                value_return = "Error to start task in VM '{}'".format(vm_name)

        elif action == Daemon.STATUS:
            try:

                value_return = self.__get_command_status(session_name)
                status_return = Daemon.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error to get VM {} status".format(vm_name)
                status_return = Daemon.ERROR

        elif action == Daemon.STOP:

            try:
                value_return = self.___stop_command(session_name, command)
                status_return = Daemon.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error stop command {} in VM {} status".format(command, vm_name)
                status_return = Daemon.ERROR

        # elif action == Daemon.INSTANCE_USAGE:
        #     try:
        #
        #         value_return = self.__get_instance_usage()
        #         status_return = Daemon.SUCCESS
        #     except Exception as e:
        #         logging.error(e)
        #         value_return = "Error to get instance {} usage".format(self.instance_id)
        #         status_return = Daemon.ERROR

        elif action == Daemon.TEST:
            value_return = "Hello world"
            status_return = Daemon.SUCCESS

        else:
            value_return = "invalid command"
            status_return = Daemon.ERROR

        duration = datetime.now() - start_time
        logging.info(str({"status": status_return, "value": value_return, "duration": str(duration)}))

        return {"status": status_return, "value": value_return, "duration": str(duration)}

    # def __get_instance_status(self, command):
    #     # check container status
    #     cmd_cpu = "top -b -n 10 -d.2 | grep 'Cpu'|  awk 'NR==3{ print($2)}'"
    #     cmd_memory = "top -b -n 10 -d.2 | grep 'Mem' |  awk 'NR==3{ print($4)}'"
    #
    #     ps = subprocess.Popen(cmd_cpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     out1 = ps.communicate()[0]
    #
    #     ps = subprocess.Popen(cmd_memory, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     out2, err = ps.communicate()[0]
    #
    #     cpu_usage = out1
    #     memory_usage = out2
    #
    #     return {
    #         "memory": memory_usage,
    #         "cpu": cpu_usage
    #     }

    def __get_command_status(self, session_name):

        # check if our screen session is still running
        cmd = "screen -S {} -Q select . ; echo $?".format(
            session_name
        )

        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()

        out = int(out.strip())

        # if cmd command return 0 it means that the screen session is still running
        if out == 0:
            status = 'running'
        # TODO: Garantir a conclusão ou a parada no meio da aplicação
        else:
            status = 'not running'

        # TODO: Pegar qual dos estagios do MASA-CUDAlign ele ta executando
        current_stage = 0

        return {"status": status, "current_stage": current_stage}

    def ___stop_command(self, session_name, command):

        operation_time = timedelta(seconds=0.0)

        values = self.__get_command_status(session_name)

        if values['status'] == 'running':

            cmd = "screen -X -S {} quit".format(session_name)

            logging.info(cmd)

            start_time = datetime.now()
            subprocess.run(cmd.split())
            end_time = datetime.now()

            operation_time = end_time - start_time

            msg = "Screen session {} that was running command '{}' stopped".format(session_name, command)

        else:
            msg = "Screen session {} with command '{}' is not running".format(session_name, command)

        return {"msg": msg, "duration": str(operation_time)}

    def __start_command(self, session_name, command):
        # start application without checkpoint

        cmd = "screen -L -Logfile {}/screen_log -S {} -dm bash -c '{}'".format(
            self.root_path, session_name, command
        )

        logging.info(cmd)

        subprocess.run(cmd.split())


class MyWebService(object):

    def __init__(self, args):
        self.daemon = Daemon(
            vm_user=args.vm_user,
            root_path=args.root_path,
            work_path=args.work_path,
            task_id=args.task_id,
            execution_id=args.execution_id,
            instance_id=args.instance_id
        )

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def process(self):
        data = cherrypy.request.json
        logging.info(data)
        return self.daemon.handle_command(action=data['action'], value=data['value'])


def main():
    parser = argparse.ArgumentParser(description='Execute GPU application with checkpoint record.')

    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--work_path', type=str, required=True)

    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--execution_id', type=int, required=True)
    parser.add_argument('--instance_id', type=str, required=True)

    parser.add_argument('--vm_user', type=str, required=True)

    args = parser.parse_args()

    config = {'server.socket_host': '0.0.0.0'}
    cherrypy.config.update(config)
    cherrypy.quickstart(MyWebService(args))

    # create a daemon


if __name__ == "__main__":
    main()
