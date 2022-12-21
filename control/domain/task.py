from control.config.checkpoint_config import CheckPointConfig

from typing import Dict


class Task:
    RESTART = 'client_restart'
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

    def __init__(self, task_id, task_name, command, generic_checkpoint, runtime=None):
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

        self.generic_checkpoint = generic_checkpoint

        if self.checkpoint_config.with_checkpoint and self.generic_checkpoint:
            self.__compute_checkpoint_values()

        self.has_checkpoint = False
        self.do_checkpoint = True

        self.server_ip = None

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
    def from_dict(cls, a_dict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(task_id),
                task_name=a_dict['tasks'][task_id]['task_name'],
                # memory=a_dict['tasks'][task_id]['memory'],
                # io=a_dict['tasks'][task_id]['io'],
                command=a_dict['tasks'][task_id]['command'],
                runtime=a_dict['tasks'][task_id]['runtime'],
                generic_checkpoint=a_dict['tasks'][task_id]['generic_checkpoint']
            )
            for task_id in a_dict['tasks']
        ]

    def __str__(self):
        return "Task_id: {}, command:{}, generic_checkpoint:{}".format(
            self.task_id,
            self.command,
            self.generic_checkpoint
        )

    def get_runtime(self, instance_type):

        if instance_type in self.runtime:
            return self.runtime[instance_type]
        else:
            return None

    def is_running(self):
        pass

    def start_execution(self, instance_type):
        pass

    def stop_execution(self):
        pass

    def finish_execution(self):
        pass

    def has_task_finished(self):
        pass

    def get_running_instance(self):
        pass

    def update_execution_time(self, time_executed):
        pass
