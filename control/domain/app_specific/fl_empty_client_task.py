from control.domain.task import Task


class FLEmptyClientTask(Task):

    def __init__(self, task_id, task_name, command, generic_checkpoint, client_id,
                 zip_file, dataset_urn):
        super().__init__(task_id, task_name, command, generic_checkpoint)

        self.simple_command = command
        self.client_id = client_id
        self.zip_file = zip_file
        self.dataset_urn = dataset_urn
        
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
        return self.finished

    def get_running_instance(self):
        return self.running_instance

    @classmethod
    def from_dict(cls, a_dict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(key),
                task_name=a_dict['tasks']['clients'][key]['task_name'],
                client_id=int(key),
                command=a_dict['tasks']['clients'][key]['command'],
                generic_checkpoint=a_dict['tasks']['clients'][key]['generic_checkpoint'],
                zip_file=a_dict['tasks']['clients'][key]['zip_file'],
                dataset_urn=a_dict['tasks']['clients'][key]['dataset_urn']
            )
            for key in a_dict['tasks']['clients']
        ]

    def __str__(self):
        return "FLTILClientTask_id: {}, command: {}, generic_checkpoint: {}, " \
               "client_id: {}".format(self.task_id,
                                      self.command,
                                      self.generic_checkpoint,
                                      self.client_id)

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen
