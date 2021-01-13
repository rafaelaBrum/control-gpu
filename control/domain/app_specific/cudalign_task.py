from control.domain.task import Task

from typing import Dict


class CUDAlignTask(Task):

    tam_cell = 8

    baseline_instance = "g4dn.xlarge"

    baseline_runtimes_relation_similar_seqs = {
        "g2.2xlarge": 3.2,
        "g3s.xlarge": 1.7,
        "g4dn.xlarge": 1.0,
        "g4dn.2xlarge": 0.9,
        "p2.xlarge": 2.8
    }

    baseline_runtimes_relation_different_seqs = {
        "g2.2xlarge": 5.0,
        "g3s.xlarge": 2.1,
        "g4dn.xlarge": 1.0,
        "g4dn.2xlarge": 1.1,
        "p2.xlarge": 4.0
    }

    def __init__(self, task_id, command, generic_ckpt, runtime, mcups, tam_seq0, tam_seq1, similar_seqs,
                 disk_size, work_dir=""):
        super().__init__(task_id, command, generic_ckpt, runtime)

        if self.baseline_instance not in self.runtime:
            raise Exception("CUDAlignTask Error: CUDAlignTask '{}' don't have run time "
                            "for instance {}".format(task_id, self.baseline_instance))

        self.mcups: Dict[str, float] = mcups

        if self.baseline_instance not in self.mcups:
            raise Exception("CUDAlignTask Error: CUDAlignTask '{}' don't have MCUPS for "
                            "instance {}".format(task_id, self.baseline_instance))

        self.tam_seq0 = tam_seq0
        self.tam_seq1 = tam_seq1

        self.similar_seqs = similar_seqs

        if disk_size.endswith('G'):
            disk_size_int = int((disk_size.split('G')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024 * 1024
        elif disk_size.endswith('M'):
            disk_size_int = int((disk_size.split('M')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024

        self.work_dir = work_dir
        self.__calculate_flush_interval()
        self.__calculate_runtimes_and_mcups()

    def __calculate_runtimes_and_mcups(self):
        if self.similar_seqs:
            iter_baselines = self.baseline_runtimes_relation_similar_seqs
        else:
            iter_baselines = self.baseline_runtimes_relation_different_seqs
        for key, value in iter_baselines.items():
            if key not in self.runtime:
                self.runtime[key] = self.runtime[self.baseline_instance]*value
            if key not in self.mcups:
                self.mcups[key] = self.mcups[self.baseline_instance]/value

    def __calculate_flush_interval(self):
        self.flush_interval = int((self.tam_seq0*self.tam_seq1*self.tam_cell)/self.disk_limit + 1)

    @classmethod
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return [
            cls(
                task_id=int(task_id),
                # memory=adict['cudalign_tasks'][task_id]['memory'],
                # io=adict['cudalign_tasks'][task_id]['io'],
                command=adict['cudalign_tasks'][task_id]['command'],
                runtime=adict['cudalign_tasks'][task_id]['runtime'],
                generic_ckpt=adict['cudalign_tasks'][task_id]['generic_ckpt'],
                disk_size=adict['cudalign_tasks'][task_id]['disk_size'],
                mcups=adict['cudalign_tasks'][task_id]['mcups'],
                tam_seq0=adict['cudalign_tasks'][task_id]['tam_seq0'],
                tam_seq1=adict['cudalign_tasks'][task_id]['tam_seq1'],
                similar_seqs=adict['cudalign_tasks'][task_id]['similar_seqs']
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

    def print_all_runtimes(self):
        screen = ""
        for key, value in sorted(self.runtime.items()):
            screen += "{}: {} s\n".format(key, value)

        return screen

    def print_all_mcups(self):
        screen = ""
        for key, value in sorted(self.mcups.items()):
            screen += "{}: {}\n".format(key, value)

        return screen
