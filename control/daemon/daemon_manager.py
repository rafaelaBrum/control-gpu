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
    # CHECKPOINT_STOP = 'checkpoint_stop'
    # RESTART = 'restart'
    STOP = 'stop'
    # CHECKPOINT = 'checkpoint'
    # TASK_USAGE = 'task_usage'
    # INSTANCE_USAGE = 'instance_usage'
    TEST = 'test'
    SUCCESS = 'success'
    ERROR = 'error'

    def __init__(self, vm_user, root_path, work_path, container_image, job_id, execution_id, instance_id):
        self.vm_user = vm_user

        self.work_path = work_path
        self.container_image = container_image

        self.job_id = job_id
        self.execution_id = execution_id
        self.instance_id = instance_id

        self.root_path = os.path.join(root_path, "{}_{}/".format(self.job_id, self.execution_id))

        self.__prepare_logging()

    # waiting for commands

    def __prepare_logging(self):

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        root_logger = logging.getLogger()
        root_logger.setLevel('INFO')

        file_name = os.path.join(self.root_path,
                                 "{}_{}_{}.log".format(self.job_id, self.execution_id, self.instance_id))

        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

    def handle_command(self, action, value):

        task_id = value['task_id']
        command = value['command']

        session_name = "Session_{}_{}_{}_{}".format(
            value['command'].split()[0],
            self.job_id,
            self.execution_id,
            task_id
        )

        vm_name = "VM_{}_{}_{}".format(
            self.job_id,
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

        # elif action == Daemon.CHECKPOINT_STOP:
        #
        #     # Try to create the backup_path
        #     cmd_folder = "mkdir -p {}".format(backup_path)
        #     subprocess.run(cmd_folder.split())
        #
        #     try:
        #         checkpoint_return = self.__checkpoint(container_name, backup_path)
        #         stop_return = self.___stop_container(container_name)
        #
        #         value_return = {"checkpoint": checkpoint_return, "stop": stop_return}
        #
        #         status_return = Daemon.SUCCESS
        #     except Exception as e:
        #         logging.error(e)
        #         value_return = "Error to checkpoint and stop container {} status".format(container_name)
        #         status_return = Daemon.ERROR

        # elif action == Daemon.RESTART:
        #
        #     # Try to create the backup_path
        #     cmd_folder = "mkdir -p {}".format(backup_path)
        #     subprocess.run(cmd_folder.split())
        #
        #     try:
        #         value_return = self.__restart_from_last_checkpoint(container_name, data_path, command, backup_path)
        #         status_return = Daemon.SUCCESS
        #     except Exception as e:
        #         value_return = "Error to restart container {} status".format(container_name)
        #         status_return = Daemon.ERROR
        #         logging.error(e)

        elif action == Daemon.STOP:

            try:
                value_return = self.___stop_command(session_name, command)
                status_return = Daemon.SUCCESS
            except Exception as e:
                logging.error(e)
                value_return = "Error stop command {} in VM {} status".format(command, vm_name)
                status_return = Daemon.ERROR

        # elif action == Daemon.CHECKPOINT:
        #
        #     # Try to create the backup_path
        #     cmd_folder = "mkdir -p {}".format(backup_path)
        #     subprocess.run(cmd_folder.split())
        #
        #     try:
        #         value_return = self.__checkpoint(container_name, backup_path)
        #         status_return = Daemon.SUCCESS
        #     except Exception as e:
        #         logging.error(e)
        #         value_return = "Error to checkpoint container {} status".format(container_name)
        #         status_return = Daemon.ERROR

        # elif action == Daemon.TASK_USAGE:
        #     try:
        #
        #         value_return = self.__get_container_usage(container_name, data_path)
        #         status_return = Daemon.SUCCESS
        #     except Exception as e:
        #         logging.error(e)
        #         value_return = "Error to get container {} usage".format(container_name)
        #         status_return = Daemon.ERROR
        #
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

    # def __copy_and_overwrite(self, from_path, to_path):
    #     # given permition to the copy
    #     cmd = "sudo chown -R {}:{} {}".format(self.vm_user, self.vm_user, from_path)
    #
    #     logging.info("Copy and overwrite: {}".format(cmd))
    #
    #     process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    #     out, err = process.communicate()
    #
    #     logging.info(out)
    #     logging.error(err)
    #
    #     if os.path.exists(to_path):
    #         shutil.rmtree(to_path)
    #
    #     shutil.copytree(from_path, to_path)

    # def __get_container_usage(self, container_name, data_path):
    #     # check container status
    #     cmd = "docker container stats {} --no-stream".format(
    #         container_name
    #     )
    #     process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    #     out, err = process.communicate()
    #
    #     line = out.decode().splitlines()[-1]
    #
    #     cpu_usage = line.split()[1]
    #     memory = line.split()[2]
    #     memory_index = line.split()[3]
    #
    #     # get super-task pos
    #     pos = 0
    #
    #     exec_bkp_file = Path(data_path, 'exec.bkp')
    #
    #     if exec_bkp_file.is_file():
    #
    #         try:
    #             with open(exec_bkp_file) as bkp_fp:
    #                 pos = int(bkp_fp.read())
    #         except Exception as e:
    #             pos = 0
    #             logging.error("error to get pos from exec.bkp")
    #             logging.error(e)
    #
    #     return {
    #         "memory": memory + " " + memory_index,
    #         "cpu": cpu_usage,
    #         "pos": pos
    #     }

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

    # def __restart_from_last_checkpoint(self, container_name, data_path, container_cmd, backup_path):
    #
    #     # execute command to load the s3 directory
    #     os.listdir(backup_path)
    #
    #     status = self.__get_container_status(container_name)["status"]
    #
    #     if status == "not found":
    #         self.__create_container(data_path=data_path,
    #                                 container_name=container_name,
    #                                 container_cmd=container_cmd)
    #
    #     checkpoint_list = self.__get_checkpoint_list(backup_path)
    #
    #     if checkpoint_list is not None:
    #
    #         created = False
    #
    #         for checkpoint_name in checkpoint_list:
    #             cmd = 'docker start --checkpoint={} --checkpoint-dir={} {}'.format(
    #                 checkpoint_name,
    #                 backup_path,
    #                 container_name
    #             )
    #
    #             logging.info(cmd)
    #
    #             subprocess.run(cmd.split())
    #
    #             # check if checkpoint was created with success
    #             info = self.__get_container_status(container_name)
    #             if info['status'] == 'running':
    #                 created = True
    #                 break
    #
    #         # check if recovery was created with success
    #         if not created:
    #             logging.error('Error to recovery container {} from checkpoint'.format(container_name))
    #             raise Exception('Error to recovery container from checkpoint')
    #
    #         msg = "container {} restarted with success".format(container_name)
    #
    #     else:
    #         # container don't have a checkpoint
    #         self.__start_container(container_name)
    #         msg = "container {} started without checkpoint".format(container_name)
    #
    #     logging.info(msg)
    #
    #     return msg

    # def __checkpoint(self, container_name, backup_path):
    #     status = self.__get_container_status(container_name)["status"]
    #
    #     # regex to extract checkpoint number from checkpoint file
    #     re_checkpoint = "[^0-9]*([0-9]*)"
    #
    #     cpcount = 0
    #
    #     code = None
    #     err = ""
    #     out = ""
    #
    #     if status == "running":
    #
    #         checkpoint_list = self.__get_checkpoint_list(backup_path)
    #
    #         if checkpoint_list is not None:
    #             newest = checkpoint_list[0]
    #             m = re.search(re_checkpoint, newest)
    #             cpcount = int(m.group(1))
    #             cpcount = (cpcount + 1) % Daemon.CHECKPOINT_LIMIT
    #
    #         checkpoint_name = 'checkpoint{}'.format(cpcount)
    #
    #         # If checkpoint already exist, we have to delete it
    #         if os.path.isdir(backup_path + checkpoint_name):
    #             rm_cmd = "sudo rm -rf {}".format(os.path.join(backup_path, checkpoint_name))
    #             logging.info(rm_cmd)
    #             subprocess.run(rm_cmd.split())
    #             # shutil.rmtree(backup_path + checkpoint_name)
    #
    #         cmd1 = "docker checkpoint create --checkpoint-dir={} --leave-running=true {} {}".format(
    #             backup_path,
    #             container_name,
    #             checkpoint_name)
    #
    #         logging.info(cmd1)
    #
    #         r = subprocess.run(cmd1.split())
    #
    #         if r.returncode != 0:
    #             err = r.stderr
    #             out = r.stdout
    #             code = r.returncode
    #
    #             msg = "Checkpoint of container {} Stderr: {} stdout: {} Code: {}".format(container_name, err, out, code)
    #
    #         else:
    #             msg = "Checkpoint of container {} created with success".format(container_name)
    #
    #     else:
    #         msg = "Checkpoint Error - Container {}  is not running".format(container_name)
    #
    #     return {"msg": msg, "code": code, "out": out, "err": err}

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

    # def __get_checkpoint_list(self, backup_path):
    #
    #     os.chdir(backup_path)
    #
    #     checkpoint_list = sorted(os.listdir(os.getcwd()), key=os.path.getmtime, reverse=True)
    #
    #     logging.info('checkpoint list size: {}'.format(len(checkpoint_list)))
    #
    #     if len(checkpoint_list) > 0:
    #         return checkpoint_list
    #
    #     return None


class MyWebService(object):

    def __init__(self, args):
        self.daemon = Daemon(
            vm_user=args.vm_user,
            root_path=args.root_path,
            work_path=args.work_path,
            container_image=args.container_image,
            job_id=args.job_id,
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
    parser = argparse.ArgumentParser(description='Execute docker image with checkpoint record.')

    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--work_path', type=str, required=True)

    parser.add_argument('--container_image', type=str, required=True)
    parser.add_argument('--job_id', type=int, required=True)
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
