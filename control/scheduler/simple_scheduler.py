from control.domain.app_specific.cudalign_task import CUDAlignTask
from control.domain.instance_type import InstanceType

from control.managers.cloud_manager import CloudManager

from typing import Dict

import logging


class SimpleScheduler:

    def __init__(self, instance_types: Dict[str, InstanceType]):
        self.instance_types_spot = instance_types
        self.instance_types_on_demand = instance_types.copy()
        self.deadline_spot = 0

    # def add_instance_type(self, name_type, instance):
    #     self.instance_types[name_type] = instance

    def remove_instance_type_spot(self, name_instance_type):
        self.instance_types_spot.pop(name_instance_type)

    def calculate_max_restart_time(self, cudalign_task: CUDAlignTask):
        max_restart_time = 0.0
        for name_instance_type, instance in self.instance_types_on_demand.items():
            max_restart_time = max(max_restart_time, (cudalign_task.get_restart_overhead(name_instance_type) +
                                   instance.boot_overhead_seconds))
        return max_restart_time

    def choose_initial_best_instance_type(self, cudalign_task: CUDAlignTask, deadline):
        self.deadline_spot = deadline - self.calculate_max_restart_time(cudalign_task)
        # logging.info("<Scheduler>: Choosing initial instance for CUDAlignTask {} "
        #              "with deadline {} (considering future restart)".format(cudalign_task.task_id,
        #                                                                     deadline_spot))
        possible_vms: Dict[str:float] = dict()
        for name_instance_type, instance in self.instance_types_spot.items():
            # logging.info("<Scheduler>: Testing spot instance {}".format(name_instance_type))
            runtime = cudalign_task.get_runtime(name_instance_type)
            # logging.info("<Scheduler>: Runtime in spot instance {}: {} s".format(name_instance_type, runtime))
            if runtime < self.deadline_spot:
                possible_vms[name_instance_type] = runtime*(instance.price_preemptible/3600)
                # logging.info("<Scheduler>: Spot instance {} can be chosen "
                #              "and will cost US${}".format(name_instance_type, possible_vms[name_instance_type]))

        if len(possible_vms) > 0:
            # order by expected cost
            ordered_possible_vms = {k: v for k, v in sorted(possible_vms.items(), key=lambda item: item[1])}
            logging.info("<Scheduler>: Spot instance chosen {}".format(list(ordered_possible_vms.keys())[0]))
            # return the type of selected VM
            selected_instance_type = list(ordered_possible_vms.keys())[0]
            return self.instance_types_spot[selected_instance_type], CloudManager.PREEMPTIBLE
        else:
            logging.error("<Scheduler>: Could not chose an spot VM!")
            # logging.info("<Scheduler>: Choosing an on-demand VM with deadline {}!".format(deadline))
            possible_vms: Dict[str:float] = dict()
            for name_instance_type, instance in self.instance_types_on_demand.items():
                # logging.info("<Scheduler>: Testing on-demand instance {}".format(name_instance_type))
                runtime = cudalign_task.get_runtime(name_instance_type)
                # logging.info("<Scheduler>: Runtime in on-demand instance {}: {}".format(name_instance_type, runtime))
                if runtime < deadline:
                    possible_vms[name_instance_type] = runtime * (instance.price_ondemand / 3600)
                    # logging.info("<Scheduler>: On-demand instance {} can be chosen "
                    #              "and will cost US${}".format(instance.type, possible_vms[name_instance_type]))

            if len(possible_vms) > 0:
                # order by expected cost
                ordered_possible_vms = {k: v for k, v in sorted(possible_vms.items(), key=lambda item: item[1])}
                logging.info("<Scheduler>: On-demand instance chosen {}".format(list(ordered_possible_vms.keys())[0]))
                # return the type of selected VM
                selected_instance_type = list(ordered_possible_vms.keys())[0]
                return self.instance_types_on_demand[selected_instance_type], CloudManager.ON_DEMAND
            else:
                logging.error("<Scheduler>: No VM could be selected!")
                return "", ""

    def choose_restart_best_instance_type(self, cudalign_task: CUDAlignTask, deadline, current_time):
        # logging.info("<Scheduler>: Choosing restart instance for CUDAlignTask {}".format(cudalign_task.task_id))
        cudalign_task.update_percentage_done()
        # logging.info("<Scheduler>: Task {} already executed {}%".format(cudalign_task.task_id,
        #                                                                 cudalign_task.percentage_executed))

        remaining_deadline = deadline - current_time
        try:
            self.remove_instance_type_spot(cudalign_task.get_running_instance())
        except Exception as e:
            logging.error(e)

        # logging.info("<Scheduler>: Remaining deadline for task {} is {}".format(cudalign_task.task_id,
        #                                                                         remaining_deadline))

        remaining_deadline_spot = self.deadline_spot - current_time
        # logging.info("<Scheduler>: Choosing restart spot instance for CUDAlignTask {} with remaining "
        #              "deadline {} (considering future restart)".format(cudalign_task.task_id,
        #                                                                remaining_deadline_spot))

        possible_vms: Dict[str:float] = dict()
        for name_instance_type, instance in self.instance_types_spot.items():
            # logging.info("<Scheduler>: Testing spot instance {}".format(name_instance_type))
            runtime = cudalign_task.get_remaining_execution_time_with_restart(name_instance_type) + \
                      instance.boot_overhead_seconds
            # logging.info("<Scheduler>: Runtime in spot instance {}: {} s".format(name_instance_type, runtime))
            if runtime < remaining_deadline_spot:
                possible_vms[name_instance_type] = runtime*instance.price_preemptible
                # logging.info("<Scheduler>: Spot instance {} can be chosen "
                #              "and will cost US${}".format(instance.type, possible_vms[name_instance_type]))

        # # add again the running instance in the types of scheduler
        # self.add_instance_type(cudalign_task.get_running_instance(), instance_removed)

        if len(possible_vms) > 0:
            # order by expected cost
            ordered_possible_vms = {k: v for k, v in sorted(possible_vms.items(), key=lambda item: item[1])}
            # logging.info("<Scheduler>: Spot instance chosen {}".format(list(ordered_possible_vms.keys())[0]))
            # return the type of selected VM
            selected_instance_type = list(ordered_possible_vms.keys())[0]
            return self.instance_types_spot[selected_instance_type], CloudManager.PREEMPTIBLE
        else:
            # logging.info("<Scheduler>: Could not chose an spot VM!")
            # logging.info("<Scheduler>: Choosing restart on-demand instance for CUDAlignTask {} "
            #              "with remaining deadline {}".format(cudalign_task.task_id, remaining_deadline))

            possible_vms: Dict[str:float] = dict()

            for name_instance_type, instance in self.instance_types_on_demand.items():
                # logging.info("<Scheduler>: Testing on-demand instance {}".format(name_instance_type))
                runtime = cudalign_task.get_remaining_execution_time_with_restart(name_instance_type)
                # logging.info("<Scheduler>: Runtime in on-demand instance {}: {}".format(name_instance_type, runtime))
                if runtime < remaining_deadline:
                    possible_vms[name_instance_type] = runtime * (instance.price_ondemand / 3600)
                    # logging.info("<Scheduler>: On-demand instance {} can be chosen "
                    #              "and will cost US${}".format(name_instance_type, possible_vms[name_instance_type]))

            if len(possible_vms) > 0:
                # order by expected cost
                ordered_possible_vms = {k: v for k, v in sorted(possible_vms.items(), key=lambda item: item[1])}
                # logging.info("<Scheduler>: On-demand instance chosen {}".format(list(ordered_possible_vms.keys())[0]))
                # return the type of selected VM
                selected_instance_type = list(ordered_possible_vms.keys())[0]
                return self.instance_types_on_demand[selected_instance_type], CloudManager.ON_DEMAND
            else:
                logging.error("<Scheduler>: No VM could be selected!")
                return "", ""
