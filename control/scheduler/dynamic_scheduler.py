from control.util.loader import Loader
from control.domain.instance_type import InstanceType

from math import inf
from copy import deepcopy
from typing import Dict

import logging
import json

from control.scheduler.mathematical_formulation_scheduler import MathematicalFormulationScheduler


class DynamicScheduler(MathematicalFormulationScheduler):

    def __init__(self, loader: Loader):
        super().__init__(loader=loader)
        self.max_cost = 0.0
        self.max_total_exec = 0.0
        self.expected_makespan = 0.0
        self.expected_cost = 0.0
        self.__read_limits_json(loader.map_file)
        self.dynamic_scheduler_instances_server_cloudlab = deepcopy(self.instances_server_cloudlab)
        self.dynamic_scheduler_instances_server_cloud = {k: v for d in (deepcopy(self.instances_server_aws),
                                                                        deepcopy(self.instances_server_gcp))
                                                         for k, v in d.items()}
        self.dynamic_scheduler_inst_cli_cloudlab: Dict[int, Dict[str, InstanceType]] = {}
        self.dynamic_scheduler_inst_cli_cloud: Dict[int, Dict[str, InstanceType]] = {}
        for cli in self.clients:
            self.dynamic_scheduler_inst_cli_cloudlab[cli] = deepcopy(self.instances_client_cloudlab)
            self.dynamic_scheduler_inst_cli_cloud[cli] = {k: v for d in (deepcopy(self.instances_client_aws),
                                                                         deepcopy(self.instances_client_gcp))
                                                          for k, v in d.items()}

    def choose_client_new_instance_cloudlab(self, client_num):
        min_greedy_value = inf
        min_greedy_makespan = inf
        min_greedy_cost = inf
        instance_chosen = None
        location_chosen = None
        server_inst = self.current_vms['server']
        server_loc = self.current_locations['server']
        current_inst = self.current_vms[str(client_num)]
        current_loc = self.current_locations[str(client_num)]

        aux_loc = current_inst.provider + '_' + current_loc.region

        # if current_inst.type in self.dynamic_scheduler_inst_cli_cloudlab[client_num]:
        #     try:
        #         if aux_loc in self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations:
        #             self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations.remove(aux_loc)
        #             logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
        #     except Exception as e:
        #         logging.error(f"<Scheduler> Error removing {aux_loc} from "
        #                       f" {self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations}")
        #         logging.error(e)
        #
        #     if not self.dynamic_scheduler_inst_cli_cloudlab[client_num][current_inst.type].locations:
        #         logging.info(f"Popping {current_inst.type} from possible future VMs")
        #         self.dynamic_scheduler_inst_cli_cloudlab[client_num].pop(current_inst.type)

        current_time = self.time_exec[client_num,
                                      current_inst.provider.upper(),
                                      current_loc.region,
                                      current_inst.type] + self.time_comm[server_inst.provider.upper(),
                                                                          server_loc.region,
                                                                          current_inst.provider.upper(),
                                                                          current_loc.region]

        if self.expected_makespan == current_time:
            self.expected_makespan = self.__get_expected_makespan(skip=True,
                                                                  client_num=client_num)

        self.expected_cost = self.__compute_new_expected_cost(skip=True,
                                                              client_num=client_num,
                                                              makespan=self.expected_makespan)

        for instance_type, instance in self.dynamic_scheduler_inst_cli_cloudlab[client_num].items():

            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_time = self.time_exec[client_num,
                                                  instance.provider.upper(),
                                                  loc,
                                                  instance.type] + \
                                   self.time_comm[server_inst.provider.upper(),
                                                  server_loc.region,
                                                  instance.provider.upper(),
                                                  loc] + \
                                   self.time_aggreg[server_inst.provider.upper(),
                                                    server_loc.region,
                                                    server_inst.type]
                    diff_time = current_time - self.expected_makespan
                    if diff_time > 0:
                        current_cost = self.__compute_new_expected_cost(skip=True,
                                                                        client_num=client_num,
                                                                        makespan=current_time)
                        current_makespan = current_time
                    else:
                        current_cost = self.expected_cost
                        current_makespan = self.expected_makespan

                    current_cost += (self.cost_vms[instance.provider.upper(), loc, instance.type]
                                     * current_makespan)
                    current_cost += ((self.server_msg_train + self.server_msg_test)
                                     * self.cost_transfer[server_inst.provider.upper()] +
                                     (self.server_msg_train + self.server_msg_test)
                                     * self.cost_transfer[instance.provider.upper()])

                    current_value = self.alpha * (current_cost / self.max_cost) \
                                     + (1 - self.alpha) * (current_makespan / self.max_total_exec)
                except Exception as e:
                    logging.error("<Scheduler> Error getting new client VM")
                    logging.error(e)
                    current_value = inf
                    current_makespan = None
                    current_cost = None
                if current_value < min_greedy_value:
                    min_greedy_value = current_value
                    min_greedy_makespan = current_makespan
                    min_greedy_cost = current_cost
                    instance_chosen = instance
                    location_chosen = loc

        self.expected_makespan = min_greedy_makespan
        self.expected_cost = min_greedy_cost

        return instance_chosen.provider, location_chosen, instance_chosen.type

    def choose_client_new_instance(self, client_num):
        min_greedy_value = inf
        min_greedy_makespan = inf
        min_greedy_cost = inf
        instance_chosen = None
        location_chosen = None
        server_inst = self.current_vms['server']
        server_loc = self.current_locations['server']
        current_inst = self.current_vms[str(client_num)]
        current_loc = self.current_locations[str(client_num)]

        aux_loc = current_inst.provider.upper() + '_' + current_loc.region

        if current_inst.type in self.dynamic_scheduler_inst_cli_cloud[client_num]:
            try:
                if aux_loc in self.dynamic_scheduler_inst_cli_cloud[client_num][current_inst.type].locations:
                    self.dynamic_scheduler_inst_cli_cloud[client_num][current_inst.type].locations.remove(aux_loc)
                    logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
            except Exception as e:
                logging.error(f"<Scheduler> Error removing {aux_loc} from "
                              f" {self.dynamic_scheduler_inst_cli_cloud[client_num][current_inst.type].locations}")
                logging.error(e)

            if not self.dynamic_scheduler_inst_cli_cloud[client_num][current_inst.type].locations:
                logging.info(f"Popping {current_inst.type} from possible future VMs")
                self.dynamic_scheduler_inst_cli_cloud[client_num].pop(current_inst.type)

        current_time = self.time_exec[client_num,
                                      current_inst.provider.upper(),
                                      current_loc.region,
                                      current_inst.type] + self.time_comm[server_inst.provider.upper(),
                                                                          server_loc.region,
                                                                          current_inst.provider.upper(),
                                                                          current_loc.region]

        if self.expected_makespan == current_time:
            self.expected_makespan = self.__get_expected_makespan(skip=True,
                                                                  client_num=client_num)

        self.expected_cost = self.__compute_new_expected_cost(skip=True,
                                                              client_num=client_num,
                                                              makespan=self.expected_makespan)

        for instance_type, instance in self.dynamic_scheduler_inst_cli_cloud[client_num].items():

            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_time = self.time_exec[client_num,
                                                  instance.provider.upper(),
                                                  loc,
                                                  instance.type] + \
                                   self.time_comm[server_inst.provider.upper(),
                                                  server_loc.region,
                                                  instance.provider.upper(),
                                                  loc] + \
                                   self.time_aggreg[server_inst.provider.upper(),
                                                    server_loc.region,
                                                    server_inst.type]
                    diff_time = current_time - self.expected_makespan
                    if diff_time > 0:
                        current_cost = self.__compute_new_expected_cost(skip=True,
                                                                        client_num=client_num,
                                                                        makespan=current_time)
                        current_makespan = current_time
                    else:
                        current_cost = self.expected_cost
                        current_makespan = self.expected_makespan

                    current_cost += (self.cost_vms[instance.provider.upper(), loc, instance.type]
                                     * current_makespan)
                    current_cost += ((self.server_msg_train + self.server_msg_test)
                                     * self.cost_transfer[server_inst.provider.upper()] +
                                     (self.server_msg_train + self.server_msg_test)
                                     * self.cost_transfer[instance.provider.upper()])

                    current_value = self.alpha * (current_cost / self.max_cost) \
                                    + (1 - self.alpha) * (current_makespan / self.max_total_exec)
                except Exception as e:
                    logging.error("<Scheduler> Error getting new client VM")
                    logging.error(e)
                    current_value = inf
                    current_makespan = None
                    current_cost = None
                if current_value < min_greedy_value:
                    min_greedy_value = current_value
                    min_greedy_makespan = current_makespan
                    min_greedy_cost = current_cost
                    instance_chosen = instance
                    location_chosen = loc

        self.expected_makespan = min_greedy_makespan
        self.expected_cost = min_greedy_cost

        return instance_chosen.provider, location_chosen, instance_chosen.type

    def choose_server_new_instance(self):
        min_greedy_value = inf
        min_greedy_makespan = inf
        min_greedy_cost = inf
        instance_chosen = None
        location_chosen = None
        current_inst = self.current_vms['server']
        current_loc = self.current_locations['server']

        aux_loc = current_inst.provider.upper() + '_' + current_loc.region

        if current_inst.type in self.dynamic_scheduler_instances_server_cloud:
            try:
                if aux_loc in self.dynamic_scheduler_instances_server_cloud[current_inst.type].locations:
                    self.dynamic_scheduler_instances_server_cloud[current_inst.type].locations.remove(aux_loc)
                    logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
            except Exception as e:
                logging.error(f"<Scheduler> Error removing {aux_loc} from "
                              f" {self.dynamic_scheduler_instances_server_cloud[current_inst.type].locations}")
                logging.error(e)
            if not self.dynamic_scheduler_instances_server_cloud[current_inst.type].locations:
                logging.info(f"Popping {current_inst.type} from possible future VMs")
                self.dynamic_scheduler_instances_server_cloud.pop(current_inst.type)

        for instance_type, instance in self.dynamic_scheduler_instances_server_cloud.items():
            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_makespan = self.__get_expected_makespan(server_vm=instance, server_loc=loc)
                    current_cost = self.__compute_new_expected_cost(makespan=current_makespan, server_vm=instance,
                                                                    server_loc=loc)

                    current_value = self.alpha * (current_cost / self.max_cost) \
                                    + (1 - self.alpha) * (current_makespan / self.max_total_exec)
                except Exception as e:
                    logging.error("<Scheduler> Error getting new server VM")
                    logging.error(e)
                    current_value = inf
                    current_makespan = None
                    current_cost = None
                if current_value < min_greedy_value:
                    min_greedy_value = current_value
                    min_greedy_makespan = current_makespan
                    min_greedy_cost = current_cost
                    instance_chosen = instance
                    location_chosen = loc

        self.expected_makespan = min_greedy_makespan
        self.expected_cost = min_greedy_cost

        return instance_chosen.provider, location_chosen, instance_chosen.type

    def choose_server_new_instance_cloudlab(self):
        min_greedy_value = inf
        min_greedy_makespan = inf
        min_greedy_cost = inf
        instance_chosen = None
        location_chosen = None
        current_inst = self.current_vms['server']
        current_loc = self.current_locations['server']

        aux_loc = current_inst.provider + '_' + current_loc.region

        # if current_inst.type in self.dynamic_scheduler_instances_server_cloudlab:
        #     try:
        #         if aux_loc in self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations:
        #             self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations.remove(aux_loc)
        #             logging.info(f"<Scheduler> Popping {aux_loc} from {current_inst.type}")
        #     except Exception as e:
        #         logging.error(f"<Scheduler> Error removing {aux_loc} from "
        #                       f" {self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations}")
        #         logging.error(e)
        #     if not self.dynamic_scheduler_instances_server_cloudlab[current_inst.type].locations:
        #         logging.info(f"Popping {current_inst.type} from possible future VMs")
        #         self.dynamic_scheduler_instances_server_cloudlab.pop(current_inst.type)

        for instance_type, instance in self.dynamic_scheduler_instances_server_cloudlab.items():
            for aux_loc in instance.locations:
                loc = aux_loc.split('_')[-1]
                try:
                    current_makespan = self.__get_expected_makespan(server_vm=instance, server_loc=loc)
                    current_cost = self.__compute_new_expected_cost(makespan=current_makespan, server_vm=instance,
                                                                    server_loc=loc)

                    current_value = self.alpha * (current_cost / self.max_cost) \
                                    + (1 - self.alpha) * (current_makespan / self.max_total_exec)
                except Exception as e:
                    logging.error("<Scheduler> Error getting new server VM")
                    logging.error(e)
                    current_value = inf
                    current_makespan = None
                    current_cost = None
                if current_value < min_greedy_value:
                    min_greedy_value = current_value
                    min_greedy_makespan = current_makespan
                    min_greedy_cost = current_cost
                    instance_chosen = instance
                    location_chosen = loc

        self.expected_makespan = min_greedy_makespan
        self.expected_cost = min_greedy_cost

        return instance_chosen.provider, location_chosen, instance_chosen.type

    def __read_limits_json(self, map_file):
        logging.info(f"<DynamicScheduler> Reading {map_file} file")

        try:
            with open(map_file) as f:
                data = f.read()
            json_data = json.loads(data)
            self.max_cost = json_data['max_cost']
            self.max_total_exec = json_data['max_total_exec']
            self.expected_makespan = json_data['makespan']
            self.expected_cost = json_data['cost']

        except Exception as e:
            logging.error(e)

    def __get_expected_makespan(self, skip=False, client_num=-1, server_vm=None, server_loc=None):
        max_makespan = -inf
        if server_vm is None:
            server_vm = self.current_vms['server']
            server_loc = self.current_locations['server'].region

        for cli in self.clients:
            if skip and cli == client_num:
                continue
            cli_instance = self.current_vms[str(cli)]
            cli_loc = self.current_locations[str(cli)].region
            try:
                current_time = self.time_exec[cli,
                                              cli_instance.provider.upper(),
                                              cli_loc,
                                              cli_instance.type] + \
                               self.time_comm[server_vm.provider.upper(),
                                              server_loc,
                                              cli_instance.provider.upper(),
                                              cli_loc] + \
                               self.time_aggreg[server_vm.provider.upper(),
                                                server_loc,
                                                server_vm.type]
            except Exception as e:
                logging.error("<Scheduler> Error getting new client VM")
                logging.error(e)
                current_time = -inf
            if current_time > max_makespan:
                max_makespan = current_time
        return max_makespan

    def __compute_new_expected_cost(self, skip=False, client_num=-1, makespan=0.0, server_vm=None, server_loc=None):

        expected_cost = 0.0

        if server_vm is None:
            server_vm = self.current_vms['server']
            server_loc = self.current_locations['server'].region

        for cli in self.clients:
            if skip and cli == client_num:
                continue
            vm_cli = self.current_vms[str(cli)]
            loc_cli = self.current_locations[str(cli)]
            expected_cost += (self.cost_vms[vm_cli.provider.upper(), loc_cli.region, vm_cli.type]
                              * makespan)
            expected_cost += ((self.server_msg_train + self.server_msg_test)
                              * self.cost_transfer[server_vm.provider.upper()] +
                              (self.server_msg_train + self.server_msg_test)
                              * self.cost_transfer[vm_cli.provider.upper()])

        expected_cost += (self.cost_vms[server_vm.provider.upper(), server_loc, server_vm.type]
                          * makespan)

        return expected_cost
