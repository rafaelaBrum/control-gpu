from control.domain.task import Task


class FLClientTask(Task):

    def __init__(self, task_id, task_name, command, generic_ckpt, runtime, bucket_name, trainset_dir, client_id,
                 zip_file, split, batch, test_dir, train_epochs, bucket_provider, bucket_region, net):
        super().__init__(task_id, task_name, command, generic_ckpt, runtime)

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
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(key),
                task_name=adict['tasks']['clients'][key]['task_name'],
                client_id=int(key),
                command=adict['tasks']['clients'][key]['command'],
                runtime=adict['tasks']['clients'][key]['runtime'],
                generic_ckpt=adict['tasks']['clients'][key]['generic_ckpt'],
                bucket_name=adict['tasks']['clients'][key]['bucket_name'],
                trainset_dir=adict['tasks']['clients'][key]['trainset_dir'],
                zip_file=adict['tasks']['clients'][key]['zip_file'],
                split=adict['tasks']['clients'][key]['split'],
                batch=adict['tasks']['clients'][key]['batch'],
                test_dir=adict['tasks']['clients'][key]['test_dir'],
                train_epochs=adict['tasks']['clients'][key]['train_epochs'],
                bucket_provider=adict['tasks']['clients'][key]['bucket_provider'],
                bucket_region=adict['tasks']['clients'][key]['bucket_region'],
                net=adict['tasks']['clients'][key]['net']
            )
            for key in adict['tasks']['clients']
        ]

    def __str__(self):
        return "FLClientTask_id: {}, command: {}, generic_checkpoint: {}, " \
               "client_id: {}".format(self.task_id,
                                      self.command,
                                      self.generic_ckpt,
                                      self.client_id)

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen
