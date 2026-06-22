from control.domain.app_specific.fl_server_task import FLServerTask
from control.domain.app_specific.fl_til_client_task import FLTILClientTask
from control.domain.app_specific.fl_empty_client_task import FLEmptyClientTask
from control.config.application_config import ApplicationConfig


class Job:

    SERVER = 'server'
    CLIENT = 'client'

    def __init__(self, job_id, job_name, job_dict, server_msg_train, server_msg_test, client_msg_train, client_msg_test,
                 description=""):

        self.app_config = ApplicationConfig()
        self.job_id = job_id
        self.job_name = job_name

        self.server_msg_train = server_msg_train
        self.server_msg_test = server_msg_test
        self.client_msg_train = client_msg_train
        self.client_msg_test = client_msg_test

        self.description = description

        self.server_task = FLServerTask.from_dict(job_dict)
        self.client_tasks = self.__load_tasks(job_dict)

    def __load_tasks(self, job_dict):
        tasks = {}

        if self.app_config.app == "TIL":
            for task in FLTILClientTask.from_dict(job_dict):
                tasks[task.client_id] = task
        elif self.app_config.app == "empty":
            for task in FLEmptyClientTask.from_dict(job_dict):
                tasks[task.client_id] = task
        else:
            print("Need reading JSON for {} FL client application.".format(self.app_config.app))
            exit()

        return tasks

    @property
    def num_clients(self):
        return len(self.client_tasks)

    @property
    def total_tasks(self):
        return len(self.client_tasks)+1

    @classmethod
    def from_dict(cls, a_dict):
        return cls(
            job_id=a_dict['job_id'],
            job_name=a_dict['job_name'],
            job_dict=a_dict,
            description=a_dict['description'],
            server_msg_train=a_dict['server_msg_train'],
            server_msg_test=a_dict['server_msg_test'],
            client_msg_train=a_dict['client_msg_train'],
            client_msg_test=a_dict['client_msg_test']
        )
