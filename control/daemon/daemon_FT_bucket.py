#!/usr/bin/env python3
import argparse

import os
import logging

import shutil

from time import sleep

COUNT_WAITING = 5000


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def put_file(source, target, item=None):

    if item is not None:
        source = os.path.join(source, item)
        target = os.path.join(target, item)

    shutil.copyfile(source, target)


def get_file(source, target, item=None):

    if item is not None:
        source = os.path.join(source, item)
        target = os.path.join(target, item)

    shutil.copyfile(source, target)


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


def send_checkpoint(file, folder_checkpoints):
    put_file(source=os.getcwd(),
             target=folder_checkpoints,
             item=file)


def check_checkpoints(args):
    path_file = "checkpoints.txt"
    while not os.path.exists(path_file):
        continue

    with open(path_file, "r") as control_file:
        count = 0
        while True:
            lines = control_file.readlines()
            if len(lines) > 0:
                logging.info(f"Sending {lines[-1]} file")
                send_checkpoint(lines[-1], args.folder_checkpoints)
                logging.info(f"Finished sending {lines[-1]} file")
                count = 0
            else:
                count = count + 1
            if count % COUNT_WAITING:
                logging.info("Sleeping")
                sleep(300)


def get_checkpoint(args):

    path_file = "name_checkpoint.txt"

    while not os.path.exists(path_file):
        continue

    with open(path_file, "r") as file:
        ckpt_file_name = file.readlines()
        ckpt_file_name = ckpt_file_name[0][:-1]

    print("ckpt_file", ckpt_file_name)

    if ckpt_file_name == "":
        return

    get_file(source=args.folder_checkpoints,
             target=os.getcwd(),
             item=ckpt_file_name)

    new_ckpt_file_name = 'weights.npz'

    cmd = f"mv {ckpt_file_name} {new_ckpt_file_name} "

    print(cmd)
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description='Execute FT application to send checkpoint to other VM.')

    parser.add_argument('--root_path', type=str, required=True)

    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--execution_id', type=int, required=True)
    parser.add_argument('--instance_id', type=str, required=True)

    parser.add_argument('--folder_checkpoints', type=str, required=True)

    parser.add_argument('--get_file', action='store_true', dest='get_file', default=False)

    args = parser.parse_args()

    prepare_logging(args)

    args.folder_checkpoints = os.path.join(args.folder_checkpoints, "{}_{}".format(args.job_id, args.execution_id))

    if args.get_file:
        get_checkpoint(args)
    else:
        mkdirs(args.folder_checkpoints)
    
    check_checkpoints(args)


if __name__ == "__main__":
    main()
