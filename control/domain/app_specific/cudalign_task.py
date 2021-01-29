from control.domain.task import Task

from typing import Dict


class CUDAlignTask(Task):

    tam_cell = 8

    baseline_instance = "g4dn.xlarge"

    finished = False

    baseline_runtimes = {
        "g2.2xlarge": 4.16,
        "g3s.xlarge": 1.95,
        "g4dn.xlarge": 1.0,
        "g4dn.2xlarge": 1.02,
        "p2.xlarge": 3.33
    }

    def __init__(self, task_id, task_name, command, generic_ckpt, runtime, mcups, seq0, seq1, tam_seq0,
                 tam_seq1, disk_size, work_dir=""):
        super().__init__(task_id, task_name, command, generic_ckpt, runtime)

        if self.baseline_instance not in self.runtime:
            raise Exception("CUDAlignTask Error: CUDAlignTask '{}' don't have run time "
                            "for instance {}".format(task_id, self.baseline_instance))

        self.mcups: Dict[str, float] = mcups

        if self.baseline_instance not in self.mcups:
            raise Exception("CUDAlignTask Error: CUDAlignTask '{}' don't have MCUPS for "
                            "instance {}".format(task_id, self.baseline_instance))

        self.seq0 = seq0
        self.seq1 = seq1

        self.simple_command = command

        self.tam_seq0 = tam_seq0
        self.tam_seq1 = tam_seq1

        self.disk_size = disk_size

        if disk_size.endswith('G'):
            disk_size_int = int((disk_size.split('G')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024 * 1024
        elif disk_size.endswith('M'):
            disk_size_int = int((disk_size.split('M')[0]))
            self.disk_limit = disk_size_int * 1024 * 1024

        self.work_dir = work_dir
        self.__calculate_runtimes_and_mcups()
        self.__calculate_worst_scenario_restart()

        self.percentage_executed = 0.0
        self.last_interruption_time = 0
        self.real_execution_time = 0
        self.running_instance = ""
        self.running = False

    def is_running(self):
        return self.running

    def update_percentage_done(self):
        runtime_current_instance = self.real_execution_time - self.last_interruption_time
        self.last_interruption_time = self.real_execution_time
        percentage_done_current_instance = runtime_current_instance / self.runtime[self.running_instance]
        self.percentage_executed += percentage_done_current_instance
        if not self.finished and self.percentage_executed >= 1:
            self.percentage_executed = 0.9

    def __calculate_runtimes_and_mcups(self):
        for key, value in self.baseline_runtimes.items():
            if key not in self.runtime:
                self.runtime[key] = self.runtime[self.baseline_instance]*value
            if key not in self.mcups:
                self.mcups[key] = self.mcups[self.baseline_instance]/value

    def __calculate_worst_scenario_restart(self):
        self.flush_interval = int((self.tam_seq0*self.tam_seq1*self.tam_cell)/self.disk_limit + 1)

        # as the CUDAlign returns millions of cell updated per second, we need to calculate
        # in function of million of cells
        self.million_cells_worst_scenario_restart = (self.flush_interval * self.tam_seq1)/1000000

    def update_execution_time(self, time_passed):
        self.real_execution_time += time_passed

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

    def get_remaining_execution_time_with_restart(self, instance_type):
        restart_overhead = self.million_cells_worst_scenario_restart/self.mcups[instance_type]
        remaining_time_chosen_instance = self.runtime[instance_type] * (1 - self.percentage_executed)
        return remaining_time_chosen_instance + restart_overhead

    def get_execution_time(self):
        return self.real_execution_time

    def get_restart_overhead(self, instance_type):
        return self.million_cells_worst_scenario_restart/self.mcups[instance_type]

    def get_running_instance(self):
        return self.running_instance

    @classmethod
    def from_dict(cls, adict):
        """return a list of tasks created from a dict"""

        return cls(
                task_id=adict['task_id'],
                task_name=adict['task_name'],
                command=adict['command'],
                runtime=adict['runtime'],
                generic_ckpt=adict['generic_ckpt'],
                disk_size=adict['disk_size'],
                mcups=adict['mcups'],
                seq0=adict['seq0'],
                seq1=adict['seq1'],
                tam_seq0=adict['tam_seq0'],
                tam_seq1=adict['tam_seq1']
            )

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
