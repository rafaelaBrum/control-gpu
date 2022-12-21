from control.util.loader import Loader
from control.domain.instance_type import InstanceType

from math import inf
from copy import deepcopy
from typing import Dict

import logging

from control.scheduler.mathematical_formulation_scheduler import MathematicalFormulationScheduler


class DynamicScheduler(MathematicalFormulationScheduler):

    def __init__(self, loader: Loader):
        super().__init__(loader=loader)
        self.dynamic_scheduler_instances_server_cloudlab = deepcopy(self.instances_server_cloudlab)
        self.dynamic_scheduler_inst_cli_cloudlab: Dict[int, Dict[str, InstanceType]] = {}
        for cli in self.clients:
            self.dynamic_scheduler_inst_cli_cloudlab[cli] = deepcopy(self.instances_client_cloudlab)

    def choose_client_new_instance(self, client_num):
        min_time_exec = inf
        instance_chosen = None
        location_chosen = None
        server_provider = self.current_vms['server'].provider
        server_loc = self.current_locations['server'].region
        current_inst = self.current_vms[str(client_num)]
        current_loc = self.current_locations[str(client_num)]

        aux_loc = current_inst.provider + '_' + current_loc.region

        if current_inst.type in self.dynamic_scheduler_inst_cli_cloudlab[client_num]:
            try:
                if aux_loc in self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations:
                    self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations.remove(aux_loc)
                    logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
            except Exception as e:
                logging.error(f"<Scheduler> Error removing {aux_loc} from "
                              f" {self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations}")
                logging.error(e)

            if not self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations:
                logging.info(f"Popping {current_inst.type} from possible future VMs")
                self.dynamic_scheduler_inst_cli_cloudlab[client_num].pop(current_inst.type)

        for instance_type, instance in self.dynamic_scheduler_inst_cli_cloudlab[client_num].items():
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
        max_client_time_exec = -inf
        max_client_location = None
        max_client_provider = None

        for cli in self.clients:
            cli_instance = self.current_vms[str(cli)]
            cli_loc = self.current_locations[str(cli)].region
            try:
                    current_time = self.time_exec[cli,
                                                  cli_instance.provider.upper(),
                                                  cli_loc,
                                                  cli_instance.type]
            except Exception as e:
                logging.error("<Scheduler> Error getting new client VM")
                logging.error(e)
                current_time = -inf
            if current_time > max_client_time_exec:
                max_client_time_exec = current_time
                max_client_location = cli_loc
                max_client_provider = cli_instance.provider.upper()

        min_server_time = inf
        instance_chosen = None
        location_chosen = None
        current_inst = self.current_vms['server']
        current_loc = self.current_locations['server']

        aux_loc = current_inst.provider + '_' + current_loc.region

        if current_inst.type in self.dynamic_scheduler_instances_server_cloudlab:
            try:
                if aux_loc in self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations:
                    self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations.remove(aux_loc)
                    logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
            except Exception as e:
                logging.error(f"<Scheduler> Error removing {aux_loc} from "
                              f" {self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations}")
                logging.error(e)

            if not self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations:
                logging.info(f"Popping {current_inst.type} from possible future VMs")
                self.dynamic_scheduler_instances_server_cloudlab.pop(current_inst.type)

        for instance_type, instance in self.dynamic_scheduler_instances_server_cloudlab.items():
            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_time = self.time_comm[instance.provider.upper(),
                                                  loc,
                                                  max_client_provider,
                                                  max_client_location] + self.time_aggreg[instance.provider.upper(),
                                                                                          loc,
                                                                                          instance.type]
                except Exception as e:
                    logging.error("<Scheduler> Error getting new client VM")
                    logging.error(e)
                    current_time = inf
                if current_time < min_server_time:
                    min_server_time = current_time
                    instance_chosen = instance
                    location_chosen = loc
        return instance_chosen.provider, location_chosen, instance_chosen.type
