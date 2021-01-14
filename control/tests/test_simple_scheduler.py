from control.domain.app_specific.cudalign_task import CUDAlignTask
from control.domain.instance_type import InstanceType

from control.scheduler.simple_scheduler import SimpleScheduler

from control.config.logging_config import LoggingConfig
from control.config.ec2_config import EC2Config

from control.managers.ec2_manager import EC2Manager

import logging
import json

from pathlib import Path


def __prepare_logging():
    """
    Set up the log format, level and the file where it will be recorded.
    """
    logging_conf = LoggingConfig()
    log_file = Path(logging_conf.path, logging_conf.log_file)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_conf.level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def create_cudalign_task():

    try:
        task = CUDAlignTask(
            task_id=2,
            command="ls",
            runtime={'g4dn.xlarge': 64.25},
            generic_ckpt=True,
            mcups={'g4dn.xlarge': 160808.882},
            disk_size='1G',
            tam_seq0=3147090,
            tam_seq1=3282708,
            similar_seqs=False
        )
        logging.info("Created task with success. Task info: {}".format(task))
        # logging.info("Runtimes CUDAlign Task {}:\n{}".format(task.task_id, task.print_all_runtimes()))
        # logging.info("MCUPS CUDAlign Task {}:\n{}".format(task.task_id, task.print_all_mcups()))
        return task
    except Exception as e:
        logging.error(e)


def update_prices(env, env_file):
    """
    get current instances prices on EC2 and update the env dictionary and also the env.json input file
    """
    ec2_conf = EC2Config()

    zone = ec2_conf.zone
    region = ec2_conf.region

    for instance in env.values():

        if instance.market_ondemand:
            instance.setup_ondemand_price(
                price=EC2Manager.get_ondemand_price(instance_type=instance.type, region=region),
                region=region
            )

        if instance.market_preemptible:
            instance.setup_preemptible_price(
                price=EC2Manager.get_preemptible_price(instance_type=instance.type, zone=zone)[0][1],
                region=region,
                zone=zone
            )

    # Update env file
    with open(env_file, "r") as jsonFile:
        data = json.load(jsonFile)

    # updating prices on env_file
    tmp = data["instances"]
    for instance_type in tmp:
        tmp[instance_type]['prices']['on-demand'] = env[instance_type].price_ondemand
        tmp[instance_type]['prices']['preemptible'] = env[instance_type].price_preemptible

    data["instances"] = tmp

    with open(env_file, "w") as jsonFile:
        json.dump(data, jsonFile, sort_keys=False, indent=4, default=str)


def read_instances_json_file(env_file):

    try:
        with open(env_file) as f:
            env_json = json.load(f)
    except Exception as e:
        logging.error("<Loader>: Error file {} ".format(env_file))
        raise Exception(e)

    env = {}

    for instance in InstanceType.from_dict(env_json):
        env[instance.type] = instance

    return env


def test_simple_scheduler():

    # deadline in seconds!
    deadline = 100
    env_file = 'input/MASA-CUDAlign/instances.json'

    __prepare_logging()

    cudalign_task = create_cudalign_task()
    instance_types = read_instances_json_file(env_file)
    # logging.info("Variable instance_types:")
    # for instance_type, instance in instance_types.items():
    #     logging.info("{}:{}".format(instance_type, instance))
    update_prices(instance_types, env_file)
    logging.info("Prices updated")
    scheduler = SimpleScheduler(instance_types)

    instance, type_market = scheduler.choose_initial_best_instance_type(cudalign_task, deadline)
    logging.info("Starting CUDAlignTask {} in {} instance {}".format(cudalign_task.task_id, type_market, instance))
    cudalign_task.start_execution(instance)
    logging.info("Instance {} stopped with 10 seconds of execution".format(instance))
    cudalign_task.update_execution_time(10)
    instance, type_market = scheduler.choose_restart_best_instance_type(cudalign_task, deadline)
    logging.info("Restarting CUDAlignTask {} in {} instance {}".format(cudalign_task.task_id, type_market, instance))
    cudalign_task.start_execution(instance)
    logging.info("Instance {} stopped with 10 seconds of execution".format(instance))
    cudalign_task.update_execution_time(10)
    instance, type_market = scheduler.choose_restart_best_instance_type(cudalign_task, deadline)
    logging.info("Restarting CUDAlignTask {} in {} instance {}".format(cudalign_task.task_id, type_market, instance))
    cudalign_task.start_execution(instance)


