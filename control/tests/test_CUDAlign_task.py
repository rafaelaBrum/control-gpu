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

    task = CUDAlignTask(
        task_id=2,
        command="ls",
        runtime={'t2.nano': 100},
        generic_ckpt=True,
        mcups={'t2.nano': 10023.23},
        disk_size='1G',
        tam_seq0=3147090,
        tam_seq1=3282708
    )

    __prepare_logging()

    logging.info("<Task {}>: Created task with success. Task info:{}".format(task.task_id, task))
