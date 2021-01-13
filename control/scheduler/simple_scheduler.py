from control.domain.app_specific.cudalign_task import CUDAlignTask
from control.domain.instance_type import InstanceType

from typing import Dict, List


class SimpleScheduler:

    instance_types: List[InstanceType]

    def __init__(self, instance_types):
        self.instance_types = instance_types

    def choose_best_instance_type(self, cudalign_task: CUDAlignTask, deadline):
        possible_vms: Dict[str:float] = []
        for instance in self.instance_types:
            runtime = cudalign_task.get_runtime(instance.type)
            if runtime < deadline:
                possible_vms[instance.type] = runtime*instance.price_preemptible

        ordered_possible_vms = {k: v for k, v in sorted(possible_vms.items(), key=lambda item: item[1])}

        for instance in self.instance_types:
            if instance.type == list(ordered_possible_vms.keys())[0]:
                return instance
