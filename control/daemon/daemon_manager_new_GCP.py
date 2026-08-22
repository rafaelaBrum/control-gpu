#!/usr/bin/env python3
from datetime import datetime
from datetime import timedelta

import cherrypy

import argparse
import subprocess

import os
import logging

import requests

METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1'


class DaemonGCP:
    # CHECKPOINT_LIMIT = 3

    START = 'start'
    STATUS = 'status'
    STOP = 'stop'
    # TASK_USAGE = 'task_usage'
    # INSTANCE_USAGE = 'instance_usage'
    TEST = 'test'
    SUCCESS = 'success'
    ERROR = 'error'
    INSTANCE_ACTION = 'instance_action'

    def __init__(self, vm_user, root_path, job_id, task_id, execution_id, instance_id, num_rounds):
        self.vm_user = vm_user

        self.job_id = job_id
        self.task_id = task_id
        self.execution_id = execution_id
        self.instance_id = instance_id

        self.num_rounds = num_rounds

        self.root_path = os.path.join(root_path, "{}_{}_{}".format(self.job_id, self.task_id, self.execution_id))

        self.__prepare_logging()

    # waiting for commands

    def __prepare_logging(self):

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel('INFO')

        file_name = os.path.join(self.root_path,
                                 "{}_{}_{}_{}.log".format(self.job_id, self.task_id,
                                                          self.execution_id, self.instance_id))

        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    def handle_command(self, action, value):

        task_id = value['task_id']
        command = value['command']
        server_ip = value['server_ip']
        cpu = value['cpu']
        gpu = value['gpu']
        command_part = value['command_part']
        session = ''

        if command is not None:
            if isinstance(command, list):
                command = command[command_part]
            session = command.split()[0]

        session_name = "Session_{}_{}_{}_{}_{}".format(
            session,
            self.job_id,
            self.task_id,
            self.execution_id,
            task_id
        )

        vm_name = "VM_{}_{}_{}_{}".format(
            self.job_id,
            self.task_id,
            self.execution_id,
            task_id
        )

        logging.info("VM {}: Action {}".format(vm_name, action))

        start_time = datetime.now()

        if action == DaemonGCP.START:

            # Starting job
            try:

                if "client" in command:
                    command = command.replace("IP_SERVER", f"{server_ip}")

                if "flwr run" in command:
                    command = command.replace("--stream", f"--stream --run-config  \"num-server-rounds={self.num_rounds}\"")
                    

                print("Final command:", command)

                self.__start_command(session_name, command, command_part)

                status_return = DaemonGCP.SUCCESS
                value_return = "VM '{}' starts task with success".format(vm_name)

            except Exception as e:
                logging.error(e)
                status_return = DaemonGCP.ERROR
                value_return = "Error to start task in VM '{}'".format(vm_name)

        elif action == DaemonGCP.STATUS:
            try:

                value_return = self.__get_command_status(session_name, server_ip, command_part)
                status_return = DaemonGCP.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error to get VM {} status".format(vm_name)
                status_return = DaemonGCP.ERROR

        elif action == DaemonGCP.INSTANCE_ACTION:
            try:

                value_return = self.___get_instance_preempted_metadata()
                status_return = DaemonGCP.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error to get VM {} status".format(vm_name)
                status_return = DaemonGCP.ERROR

        elif action == DaemonGCP.STOP:

            try:
                value_return = self.___stop_command(session_name, command, server_ip, command_part)
                status_return = DaemonGCP.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error stop command {} in VM {} status".format(command, vm_name)
                status_return = DaemonGCP.ERROR

        elif action == DaemonGCP.TEST:
            value_return = "Hello world"
            status_return = DaemonGCP.SUCCESS

        else:
            value_return = "invalid command"
            status_return = DaemonGCP.ERROR

        duration = datetime.now() - start_time
        logging.info(str({"status": status_return, "value": value_return, "duration": str(duration)}))

        return {"status": status_return, "value": value_return, "duration": str(duration)}

    def __get_command_status(self, session_name, server_ip, command_part):

        # check if our screen session is still running
        cmd = f"screen -list | grep {session_name}"

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out_session, err = process.communicate()

        test_session = str.encode(session_name)

        if server_ip is None:
            #server task
            if test_session in out_session:
                status = 'running'
            else:
                search_string = 'Strategy execution finished'
                cmd = f"cat {self.root_path}/screen_task_log_{command_part} | grep '{search_string}'"
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                out, err = process.communicate()

                test = str.encode(search_string)

                if test in out:
                    status = 'finished'
                else:
                    status = 'not running'
            
            search_string = '\\[ROUND'
            cmd = f"cat {self.root_path}/screen_task_log_{command_part} | grep '{search_string}'"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = process.communicate()

            print(out.decode('utf-8'))

            out = out.decode('utf-8')

            rounds = 0
            
            print(out.split('\n'))
            rounds = len(out.split('\n')) - 1

            print("Rounds = ", rounds)
        else:
            #client task
            search_string = 'Sent successfully'
            cmd = f"cat {self.root_path}/screen_task_log_{command_part} | grep '{search_string}'"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = process.communicate()

            print(out.decode('utf-8'))

            out = out.decode('utf-8')

            rounds = 0

            if search_string in out:
                print(out.split('\n'))
                rounds = len(out.split('\n')) - 1

                print("Rounds = ", rounds)

            if rounds == (2 * self.num_rounds):
                status = 'finished'
            elif test_session in out_session:
                status = 'running'
            else:
                status = 'not running'

        current_stage = 0

        return {"status": status, "current_stage": current_stage, "rounds": rounds}

    def ___stop_command(self, session_name, command, server_ip, command_part):

        operation_time = timedelta(seconds=0.0)

        values = self.__get_command_status(session_name, server_ip, command_part)

        if values['status'] == 'running':

            cmd = "screen -X -S {} quit".format(session_name)

            logging.info(cmd)

            start_time = datetime.now()
            subprocess.run(cmd.split())
            end_time = datetime.now()

            operation_time = end_time - start_time

            while values['status'] == 'running':
                values = self.__get_command_status(session_name, server_ip, command_part)

            msg = "Screen session {} that was running command '{}' stopped".format(session_name, command)

        else:
            msg = "Screen session {} with command '{}' is not running".format(session_name, command)

        return {"msg": msg, "duration": str(operation_time), "rounds":values["rounds"]}

    def __start_command(self, session_name, command, command_part):
        # start application without checkpoint

        # Get PATH and LD_LIBRARY_PATH environment variables
        path = os.getenv('PATH')

        path = path + ":/home/sa_109649273287045369425/.local/bin"

        # Set PATH and LD_LIBRARY_PATH environment variables to see cudalign
        os.environ['PATH'] = path

        logging.info("PATH env: {}".format(os.getenv('PATH')))

        cmd = "screen -L -Logfile {}/screen_task_log_{} -S {} -dm bash -c {}".format(
            self.root_path, command_part, session_name, command
        )

        logging.info(cmd)

        split_cmd = cmd.split()

        arg_c_screen = split_cmd[9]

        for com in split_cmd[10:]:
            arg_c_screen = arg_c_screen + " " + com

        final_cmd = split_cmd[:9]
        final_cmd.append(arg_c_screen)

        logging.info(final_cmd)

        subprocess.run(final_cmd)

    def ___get_instance_preempted_metadata(self):
        url = '{}/instance/preempted'.format(METADATA_URL)
        r = requests.get(url, headers=METADATA_HEADERS)
        if r.status_code == 503:  # Metadata server unavailable
            logging.error('Metadata server unavailable.')
            return
        r.raise_for_status()
        return r.text


class MyWebService(object):

    def __init__(self, args):
        self.daemon = DaemonGCP(
            vm_user=args.vm_user,
            root_path=args.root_path,
            job_id=args.job_id,
            task_id=args.task_id,
            execution_id=args.execution_id,
            instance_id=args.instance_id,
            num_rounds=args.num_rounds
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

    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--execution_id', type=int, required=True)
    parser.add_argument('--instance_id', type=str, required=True)

    parser.add_argument('--vm_user', type=str, required=True)
    parser.add_argument('--socket_port', type=str, required=True)

    parser.add_argument('--num_rounds', type=int, required=True)

    args = parser.parse_args()

    config = {'server.socket_host': '0.0.0.0', 'server.socket_port': int(args.socket_port)}
    cherrypy.config.update(config)
    cherrypy.quickstart(MyWebService(args))

    # create a daemon


if __name__ == "__main__":
    main()
