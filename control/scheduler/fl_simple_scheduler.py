from control.scheduler.scheduler import Scheduler

from control.domain.instance_type import InstanceType
from control.domain.cloud_region import CloudRegion

from typing import Dict

import logging


class FLSimpleScheduler(Scheduler):

    def __init__(self, instance_types: Dict[str, InstanceType], locations: Dict[str, CloudRegion]):
        super().__init__(instance_types=instance_types, locations=locations)
