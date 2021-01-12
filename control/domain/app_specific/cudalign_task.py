from control.domain.task import Task

from typing import Dict


class CUDAlignTask(Task):

    tam_cell = 8

    base_line_runtimes_relation_similar_seqs = {
        "g2.2xlarge": 3.2,
        "g3s.xlarge": 1.7,
        "g4dn.xlarge": 1.0,
        "g4dn.2xlarge": 0.9,
        "p2.xlarge": 2.8
    }

    base_line_runtimes_relation_different_seqs = {
        "g2.2xlarge": 5.0,
        "g3s.xlarge": 2.1,
        "g4dn.xlarge": 1.0,
        "g4dn.2xlarge": 1.1,
        "p2.xlarge": 4.0
    }

    def __init__(self, task_id, command, generic_ckpt, runtime, mcups, tam_seq0, tam_seq1, disk_size, work_dir=""):
        super().__init__(task_id, command, generic_ckpt, runtime)

        self.mcups: Dict[str, float] = mcups

        self.tam_seq0 = tam_seq0
        self.tam_seq1 = tam_seq1
        if disk_size.endswith('G'):
            disk_size_int = int((disk_size.split('G')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024 * 1024
        elif disk_size.endswith('M'):
            disk_size_int = int((disk_size.split('M')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024

        self.work_dir = work_dir
        self.__calcule_flush_interval()

    def __calcule_flush_interval(self):
        self.flush_interval = int((self.tam_seq0*self.tam_seq1*self.tam_cell)/self.disk_limit + 1)

    @classmethod
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(task_id),
                # memory=adict['tasks'][task_id]['memory'],
                # io=adict['tasks'][task_id]['io'],
                command=adict['tasks'][task_id]['command'],
                runtime=adict['tasks'][task_id]['runtime'],
                generic_ckpt=adict['tasks'][task_id]['generic_ckpt'],
                disk_size=adict['tasks'][task_id]['disk_size'],
                mcups=adict['tasks'][task_id]['mcups'],
                tam_seq0=adict['tasks'][task_id]['tam_seq0'],
                tam_seq1=adict['tasks'][task_id]['tam_seq1']
            )
            for task_id in adict['tasks']
        ]

    def __str__(self):
        return "CUDAlignTask_id: {}, command:{}, generic_checkpoint:{}, " \
               "tam_seq0:{}, tam_seq1:{}, disk_limit:{}, flush_interval:{}".format(
                                                                self.task_id,
                                                                self.command,
                                                                self.generic_ckpt,
                                                                self.tam_seq0,
                                                                self.tam_seq1,
                                                                self.disk_limit,
                                                                self.flush_interval
                )
