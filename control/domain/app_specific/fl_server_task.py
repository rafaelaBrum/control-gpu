from control.domain.task import Task


class FLServerTask(Task):

    def __init__(self, task_id, task_name, command, generic_checkpoint, runtime, n_clients, n_rounds, zip_file):
        super().__init__(task_id, task_name, command, generic_checkpoint, runtime)

        self.simple_command = command

        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.zip_file = zip_file

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
    def from_dict(cls, a_dict):
        """return a list of tasks created from a dict"""

        return cls(
            task_id=int(a_dict['tasks']['server']['n_clients']),
            task_name=a_dict['tasks']['server']['task_name'],
            command=a_dict['tasks']['server']['command'],
            runtime=a_dict['tasks']['server']['runtime'],
            generic_checkpoint=a_dict['tasks']['server']['generic_checkpoint'],
            n_clients=a_dict['tasks']['server']['n_clients'],
            n_rounds=a_dict['tasks']['server']['n_rounds'],
            zip_file=a_dict['tasks']['server']['zip_file']
        )

    def __str__(self):
        return "FLServerTask_id: {}, command: {}, generic_checkpoint: {}".format(self.task_id,
                                                                                 self.command,
                                                                                 self.generic_checkpoint)

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen
