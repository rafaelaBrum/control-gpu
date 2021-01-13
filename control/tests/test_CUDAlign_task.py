from control.domain.app_specific.cudalign_task import CUDAlignTask

from control.config.logging_config import LoggingConfig

import logging

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


def test_cudalign_task_creation():

    __prepare_logging()

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
        logging.info("Runtimes CUDAlign Task {}:\n{}".format(task.task_id, task.print_all_runtimes()))
        logging.info("MCUPS CUDAlign Task {}:\n{}".format(task.task_id, task.print_all_mcups()))
    except Exception as e:
        logging.error(e)
