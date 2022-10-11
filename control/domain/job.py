from control.domain.app_specific.fl_server_task import FLServerTask
from control.domain.app_specific.fl_client_task import FLClientTask


class Job:

    SERVER = 'server'
    CLIENT = 'client'

    def __init__(self, job_id, job_name, job_dict, description=""):
        self.job_id = job_id
        self.job_name = job_name
        self.description = description

        self.server_task = FLServerTask.from_dict(job_dict)
        self.client_tasks = self.__load_tasks(job_dict)

    def __load_tasks(self, job_dict):
        tasks = {}

        for task in FLClientTask.from_dict(job_dict):
            tasks[task.client_id] = task

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
            description=a_dict['description']
        )
