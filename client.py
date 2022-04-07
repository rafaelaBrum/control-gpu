#!/usr/bin/env python
from control.util.loader import Loader

from control.scheduler.schedule_manager import ScheduleManager

from distutils.util import strtobool

from control.util.recreate_database import RecreateDatabase

from control.pre_scheduling.pre_scheduling_manager import PreSchedulingManager

import logging
import argparse


def __call_control(loader: Loader):
    try:
        loader.print_execution_info()

        manager = ScheduleManager(loader=loader)

        manager.start_execution()

        # status = "SUCCESS"

    except Exception as e:
        logging.error(e)
        # status = "ERROR"

    # if loader.dump:
    #     logging.info("Backup Database..")
    #     dump.dump_db()
    #     logging.info("Backup finished...")


def __call_pre_scheduling(loader: Loader):
    try:
        loader.print_execution_info()

        pre_sched = PreSchedulingManager(loader=loader)

        if pre_sched.stop_execution:
            return

        pre_sched.calculate_rtt_values()

        pre_sched.get_first_rounds_times()

        pre_sched.calculate_rpc_times()

        if loader.num_clients_pre_sched > 1:
            pre_sched.calculate_concurrent_rpc_times()

        pre_sched.write_json()

        # status = "SUCCESS"

    except Exception as e:
        logging.error(e)
        # status = "ERROR"

    # if loader.dump:
    #     logging.info("Backup Database..")
    #     dump.dump_db()
    #     logging.info("Backup finished...")


def __call_recreate_database(loader: Loader):
    logging.info("Are you sure that you want to recreate the database  {}? (yes or no):"
                 "".format(loader.database_conf.database_name))
    answer = input()

    try:
        if strtobool(answer):
            RecreateDatabase.execute()
            logging.info("Database was recreated with success")
        else:
            logging.error("Answer should be: yes or no")
            logging.info("Database WAS NOT recreated.")
    except Exception as e:
        logging.error(e)


def __print_execution_info(loader: Loader):
    loader.print_execution_info()


def main():
    parser = argparse.ArgumentParser(description='Multi-FedLS - v. 0.0.1')

    parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
    parser.add_argument('--job_file', help="job file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    parser.add_argument('--loc_file', help="loc file name", type=str, default=None)
    parser.add_argument('--pre_file', help="pre scheduling file name", type=str, default=None)
    # parser.add_argument('--map_file', help="map file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
    # parser.add_argument('--ac_size_seconds', help="Define the size of the Logical Allocation Cycle (seconds)",
    #                     type=int, default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)

    # parser.add_argument('--resume_rate', help="Resume rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)",
    #                     type=float, default=None)
    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)
    parser.add_argument('--num_clients_pre_sched', help="Quantity of clients in the pre-scheduling RPC tests",
                        type=int, default=None)

    # parser.add_argument('--scheduler_name',
    #                     help="Scheduler name - Currently supported Schedulers are: " + ", ".join(
    #                         Scheduler.scheduler_names),
    #                     type=str, default=None)

    # parser.add_argument('--notify', help='Send an email to notify the end of the execution (control mode)',
    #                     action='store_true', default=False)

    parser.add_argument('--server_provider', help="Server provider", type=str, default=None, required=True)
    parser.add_argument('--server_region', help="Server region", type=str, default=None, required=True)

    options_map = {
        'control': __call_control,
        'pre': __call_pre_scheduling,
        'recreate_db': __call_recreate_database,
        'info': __print_execution_info,
    }
    parser.add_argument('command', choices=options_map.keys())

    loader = Loader(args=parser.parse_args())

    func = options_map[loader.client_command]

    func(loader=loader)


if __name__ == "__main__":
    main()
