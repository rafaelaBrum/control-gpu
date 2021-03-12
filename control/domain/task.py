from control.config.checkpoint_config import CheckPointConfig

from typing import Dict


class Task:
    EXECUTING = 'executing'
    FINISHED = 'finished'
    WAITING = 'waiting'
    ERROR = 'error'
    RUNTIME_ERROR = 'runtime_error'
    MIGRATED = 'migrated'
    # HIBERNATED = 'hibernated'
    # STOLEN = 'stolen'
    # STOP_SIGNAL = 'stop_signal'

    INTERRUPTED = 'interrupted'

    # RESTARTED = 'restarted'

    def __init__(self, task_id, task_name, command, generic_ckpt, runtime):
        self.task_id = task_id
        self.task_name = task_name
        # self.memory = memory
        self.command = command
        # self.io = io
        self.runtime: Dict[str, float] = runtime

        self.checkpoint_config = CheckPointConfig()

        self.checkpoint_factor = 0.0
        self.checkpoint_number = 0
        self.checkpoint_interval = 0.0
        self.checkpoint_dump = 0.0
        self.checkpoint_overhead = 0.0

        self.generic_ckpt = generic_ckpt

        if self.checkpoint_config.with_checkpoint and self.generic_ckpt:
            self.__compute_checkpoint_values()

        self.has_checkpoint = False
        self.do_checkpoint = True

    def __compute_checkpoint_values(self):

        self.checkpoint_factor = 0.0
        self.checkpoint_number = 0
        self.checkpoint_interval = 0.0
        self.checkpoint_dump = 0.0
        self.checkpoint_overhead = 0.0
        # # get max_runtime of the tasks
        # max_runtime = max([time for time in self.runtime.values()])
        #
        # # get checkpoint factor define by the user
        # self.checkpoint_factor = self.checkpoint_config.overhead_factor
        # # computing checkpoint overhead
        # self.checkpoint_overhead = self.checkpoint_factor * max_runtime
        #
        # # computing checkpoint dump_time
        # self.checkpoint_dump = 12.99493 + 0.04 * self.memory
        #
        # # define checkpoint number
        # self.checkpoint_number = int(math.floor(self.checkpoint_overhead / self.checkpoint_dump))
        #
        # # check if checkpoint number is != 0
        # if self.checkpoint_number > 0:
        #     # define checkpoint interval
        #     self.checkpoint_interval = math.floor(max_runtime / self.checkpoint_number)
        # else:
        #     # since there is no checkpoint to take (checkpoint_number = 0) the overhead is set to zero
        #     self.checkpoint_overhead = 0.0

    @classmethod
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(task_id),
                task_name=adict['tasks'][task_id]['task_name'],
                # memory=adict['tasks'][task_id]['memory'],
                # io=adict['tasks'][task_id]['io'],
                command=adict['tasks'][task_id]['command'],
                runtime=adict['tasks'][task_id]['runtime'],
                generic_ckpt=adict['tasks'][task_id]['generic_ckpt']
            )
            for task_id in adict['tasks']
        ]

    def __str__(self):
        return "Task_id: {}, command:{}, generic_checkpoint:{}".format(
            self.task_id,
            self.command,
            self.generic_ckpt
        )

    def get_runtime(self, instance_type):

        if instance_type in self.runtime:
            return self.runtime[instance_type]
        else:
            return None
