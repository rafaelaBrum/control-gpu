from control.util.loader import Loader

from math import inf

import logging

from control.scheduler.mathematical_formulation_scheduler import MathematicalFormulationScheduler


class DynamicScheduler(MathematicalFormulationScheduler):

    def __init__(self, loader: Loader):
        super().__init__(loader=loader)

    def choose_new_client_instance(self, client_num):
        min_time_exec = inf
        instance_chosen = None
        location_chosen = None
        server_provider = self.current_vms['server'].provider
        server_loc = self.current_locations['server'].region
        current_instance = self.current_vms[str(client_num)]
        for instance in self.instances_client_cloudlab.values():
            print("current_instance", current_instance)
            print("instance.type", instance.type)
            if instance.type == current_instance:
                self.instances_client_cloudlab.pop(current_instance.type)
            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_time = self.time_exec[client_num,
                                                  instance.provider.upper(),
                                                  loc,
                                                  instance.type] + self.time_comm[server_provider.upper(),
                                                                                  server_loc,
                                                                                  instance.provider.upper(),
                                                                                  loc]
                except Exception as e:
                    logging.error("<Scheduler> Error getting new client VM")
                    logging.error(e)
                    current_time = inf
                if current_time < min_time_exec:
                    min_time_exec = current_time
                    instance_chosen = instance
                    location_chosen = loc
        return instance_chosen.provider, location_chosen, instance_chosen.type

    def choose_server_new_instance(self):
        logging.error("<Scheduler> Missing implementation of choosing server instance!")
