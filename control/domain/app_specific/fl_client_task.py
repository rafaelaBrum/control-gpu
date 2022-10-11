from control.domain.task import Task


class FLClientTask(Task):

    def __init__(self, task_id, task_name, command, generic_checkpoint, bucket_name, trainset_dir, client_id,
                 zip_file, split, batch, test_dir, train_epochs, bucket_provider, bucket_region, net):
        super().__init__(task_id, task_name, command, generic_checkpoint)

        self.simple_command = command

        self.bucket_provider = bucket_provider
        self.bucket_region = bucket_region
        self.bucket_name = bucket_name
        self.trainset_dir = trainset_dir
        self.client_id = client_id
        self.zip_file = zip_file
        self.split = split
        self.batch = batch
        self.test_dir = test_dir
        self.train_epochs = train_epochs
        self.net = net

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
                bucket_name=a_dict['tasks']['clients'][key]['bucket_name'],
                trainset_dir=a_dict['tasks']['clients'][key]['trainset_dir'],
                zip_file=a_dict['tasks']['clients'][key]['zip_file'],
                split=a_dict['tasks']['clients'][key]['split'],
                batch=a_dict['tasks']['clients'][key]['batch'],
                test_dir=a_dict['tasks']['clients'][key]['test_dir'],
                train_epochs=a_dict['tasks']['clients'][key]['train_epochs'],
                bucket_provider=a_dict['tasks']['clients'][key]['bucket_provider'],
                bucket_region=a_dict['tasks']['clients'][key]['bucket_region'],
                net=a_dict['tasks']['clients'][key]['net']
            )
            for key in a_dict['tasks']['clients']
        ]

    def __str__(self):
        return "FLClientTask_id: {}, command: {}, generic_checkpoint: {}, " \
               "client_id: {}".format(self.task_id,
                                      self.command,
                                      self.generic_checkpoint,
                                      self.client_id)

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen
