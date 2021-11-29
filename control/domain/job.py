from control.domain.app_specific.fl_server_task import FLServerTask
from control.domain.app_specific.fl_client_task import FLClientTask


class Job:

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

    @classmethod
    def from_dict(cls, adict):
        return cls(
            job_id=adict['job_id'],
            job_name=adict['job_name'],
            job_dict=adict,
            description=adict['description']
        )
