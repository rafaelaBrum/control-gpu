from control.domain.task import Task


class FLServerTask(Task):

    def __init__(self, task_id, task_name, command, generic_ckpt, runtime, n_clients, n_rounds):
        super().__init__(task_id, task_name, command, generic_ckpt, runtime)

        self.simple_command = command

        self.n_clients = n_clients
        self.n_rounds = n_rounds

        self.running_instance = ""
        self.running = False
        self.finished = False

    def is_running(self):
        return self.running

    def start_execution(self, instance_type):
        self.running_instance = instance_type
        self.running = True

    def stop_execution(self):
        self.running = False

    def finish_execution(self):
        self.finished = True
        self.running = False

    def has_task_finished(self):
        return self.finished is True

    def get_running_instance(self):
        return self.running_instance

    @classmethod
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return cls(
            task_id=adict['tasks']['server']['task_id'],
            task_name=adict['tasks']['server']['task_name'],
            command=adict['tasks']['server']['command'],
            runtime=adict['tasks']['server']['runtime'],
            generic_ckpt=adict['tasks']['server']['generic_ckpt'],
            n_clients=adict['tasks']['server']['n_clients'],
            n_rounds=adict['tasks']['server']['n_rounds']
        )

    def __str__(self):
        return "FLServerTask_id: {}, command: {}, generic_checkpoint: {}".format(self.task_id,
                                                                                 self.command,
                                                                                 self.generic_ckpt)

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen
