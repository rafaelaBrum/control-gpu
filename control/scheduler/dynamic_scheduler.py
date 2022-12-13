from control.util.loader import Loader

from math import inf

import logging

from control.scheduler.mathematical_formulation_scheduler import MathematicalFormulationScheduler


class DynamicScheduler(MathematicalFormulationScheduler):

    def __init__(self, loader: Loader):
        super().__init__(loader=loader)

    def choose_client_new_instance(self, client_num):
        min_time_exec = inf
        instance_chosen = None
        location_chosen = None
        server_provider = self.current_vms['server'].provider
        server_loc = self.current_locations['server'].region
        current_instance = self.current_vms[str(client_num)]
        current_location = self.current_locations[str(client_num)]

        aux_location = current_instance.provider + '_' + current_location.region

        if current_instance.type in self.instances_client_cloudlab:
            try:
                if aux_location in self.instances_client_cloudlab[current_instance.type].locations:
                    self.instances_client_cloudlab[current_instance.type].locations.remove(aux_location)
                    logging.info(f"<Scheduler> Popping {aux_location} from {current_instance.type}")
            except Exception as e:
                logging.error(f"<Scheduler> Error removing {aux_location} "
                              f"from {self.instances_client_cloudlab[current_instance.type].locations}")

            if not self.instances_client_cloudlab[current_instance.type].locations:
                logging.info(f"Popping {current_instance.provider + '_' + current_location} from {current_instance.type} from possible future VMs")
                self.instances_client_cloudlab.pop(current_instance.type)

        for instance_type, instance in self.instances_client_cloudlab.items():
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
