#!/usr/bin/env python3
import argparse

import os
import logging

import paramiko

import socket

from time import sleep

FOLDER_CHECKPOINTS = 'checkpoints/'


class SSHClient:

    def __init__(self, ip_address, key_path, key_file, user):

        self.ip_address = ip_address

        self.key = paramiko.Ed25519Key.from_private_key_file(key_path + key_file)
        self.user = user
        self.port = 22
        self.repeat = 5
        self.connection_timeout = 30
        self.retry_interval = 10

        self.client = None

    """
    This will check if the connection is still available.

    Return (bool) : True if it's still alive, False otherwise.
    """

    @property
    def is_active(self):

        # Check if client was initiated
        if self.client is None:
            return False

        try:
            self.client.exec_command('ls', timeout=30)
            return True
        except Exception as e:
            logging.error("<SSH Client>: Connection lost : " + str(e))
            return False

    '''
    Open a ssh connection
    Return (bool): True if the connection was open, False otherwise
    '''

    def open_connection(self):

        if not self.is_active:

            self.client = paramiko.SSHClient()
            # self.client.load_system_host_keys()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            for x in range(self.repeat):

                try:
                    self.client.connect(
                        hostname=self.ip_address,
                        port=self.port,
                        username=self.user,
                        pkey=self.key,
                        timeout=self.connection_timeout
                    )

                    tr = self.client.get_transport()
                    tr.default_max_packet_size = 100000000
                    tr.default_window_size = 100000000
                    return True

                except (paramiko.BadHostKeyException, paramiko.AuthenticationException,
                        paramiko.SSHException, socket.error) as e:

                    logging.info("<SSH Client>:" + str(x) + "> " + str(e))

                    sleep(self.retry_interval)
        else:
            logging.warning("<SSH Client>: Connection was already activated")

        return False

    '''
    close the current ssh connection
    Return (bool):  True if it was closed, False otherwise
    '''

    def close_connection(self):

        try:
            self.client.close()
            return True
        except Exception as e:
            logging.error("<SSH Client>: closing connection error " + str(e))
            return False

    def put_file(self, source, target, item=None):

        ftp_client = self.client.open_sftp()

        if item is not None:
            source = os.path.join(source, item)
            target = os.path.join(target, item)

        ftp_client.put(source, target)

        ftp_client.close()


def prepare_logging(args):

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel('INFO')

    root_path = os.path.join(args.root_path, "{}_{}_{}".format(args.job_id, args.task_id, args.execution_id))

    file_name = os.path.join(root_path,
                             "{}_{}_{}_{}_FT.log".format(args.job_id, args.task_id,
                                                         args.execution_id, args.instance_id))

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def send_checkpoint(ssh_client, file):
    if ssh_client.open_connection():
        ssh_client.put_file(source=file,
                            target=FOLDER_CHECKPOINTS)


def check_checkpoints(args):
    ssh_client = SSHClient(ip_address=args.extra_address,
                           key_path=args.key_path,
                           key_file=args.key_file,
                           user=args.user)
    with open("checkpoints.txt", "r") as control_file:
        while True:
            lines = control_file.readlines()
            if len(lines) > 0:
                send_checkpoint(ssh_client, lines[-1])


def main():
    parser = argparse.ArgumentParser(description='Execute FT application to send checkpoint to other VM.')

    parser.add_argument('--root_path', type=str, required=True)

    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--execution_id', type=int, required=True)
    parser.add_argument('--instance_id', type=str, required=True)

    parser.add_argument('--extra_address', type=str, required=True)
    parser.add_argument('--key_file', type=str, required=True)
    parser.add_argument('--key_path', type=str, required=True)
    parser.add_argument('--user', type=str, required=True)

    args = parser.parse_args()

    prepare_logging(args)
    
    check_checkpoints(args)


if __name__ == "__main__":
    main()