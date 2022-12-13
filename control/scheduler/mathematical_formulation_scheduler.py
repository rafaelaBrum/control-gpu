import os.path

from control.util.loader import Loader

import gurobipy as gp
from gurobipy import GRB
from math import inf

import logging
# import os
import json

from control.scheduler.scheduler import Scheduler


class MathematicalFormulationScheduler(Scheduler):

    def pre_process(self):

        for client in self.clients:
            for prov, region, vm in self.prov_regions_vms:
                # if self.gpu_vms[prov, region, vm] == 0:
                #     continue
                aux = (client, prov, region, vm)
                # print(aux)
                try:
                    aux_loc = self.location_ds_clients[client]
                    self.time_exec[aux] = self.baseline_exec[client] * self.slowdown[aux_loc][prov, region, vm]
                    self.client_prov_regions_vms.append(aux)
                except Exception as e:
                    logging.error(e)
                    logging.error(f"We do not support the location of client {client}: "
                                  f"{aux}")
                    # return
        self.client_prov_regions_vms = gp.tuplelist(self.client_prov_regions_vms)

        for pair in self.pair_regions:
            self.time_comm[pair] = self.comm_baseline * self.comm_slowdown[pair]

    def __init__(self, loader: Loader):
        super().__init__(instance_types=loader.env, locations=loader.loc)
        self.clients = []
        self.location_ds_clients = {}
        for client in loader.job.client_tasks.values():
            aux = client.client_id
            self.clients.append(aux)
            self.location_ds_clients[aux] = client.bucket_provider.upper() + "_" + client.bucket_region
        self.clients = gp.tuplelist(self.clients)

        # self.baseline_exec = baseline_exec
        self.baseline_exec = {}
        # self.comm_baseline = comm_baseline
        self.comm_baseline = -1
        self.server_msg_train = float(loader.job.server_msg_train)
        self.server_msg_test = float(loader.job.server_msg_test)
        self.client_msg_train = float(loader.job.client_msg_train)
        self.client_msg_test = float(loader.job.client_msg_test)
        self.T_round = float(loader.mapping_conf.deadline) / loader.job.server_task.n_rounds
        self.B_round = float(loader.mapping_conf.budget) / loader.job.server_task.n_rounds
        self.alpha = float(loader.mapping_conf.alpha)

        self.providers = []
        self.global_cpu_limits = {}
        self.global_gpu_limits = {}
        self.cost_transfer = {}
        self.prov_regions = []
        self.regional_cpu_limits = {}
        self.regional_gpu_limits = {}

        for region_id, region in loader.loc.items():
            aux = region.provider.upper()
            aux_global_cpu_limit = region.global_cpu_limit
            if aux_global_cpu_limit < 0:
                aux_global_cpu_limit = inf
            aux_global_gpu_limit = region.global_gpu_limit
            if aux_global_gpu_limit < 0:
                aux_global_gpu_limit = inf
            test_prov = self.providers.count(aux)
            if test_prov > 0:
                if self.global_cpu_limits[aux] != aux_global_cpu_limit:
                    raise Exception(f"Global vCPUs limit different in region {region_id}")
                if self.global_gpu_limits[aux] != aux_global_gpu_limit:
                    raise Exception(f"Global GPUs limit different in region {region_id}")
                if self.cost_transfer[aux] != region.cost_transfer:
                    raise Exception(f"Cost transfer different in region {region_id}")
            else:
                self.providers.append(aux)
                self.global_cpu_limits[aux] = aux_global_cpu_limit
                self.global_gpu_limits[aux] = aux_global_gpu_limit
                self.cost_transfer[aux] = region.cost_transfer
            aux = (aux, region.region)
            self.prov_regions.append(aux)
            self.regional_gpu_limits[aux] = region.regional_gpu_limit
            if self.regional_gpu_limits[aux] < 0:
                self.regional_gpu_limits[aux] = inf
            self.regional_cpu_limits[aux] = region.regional_cpu_limit
            if self.regional_cpu_limits[aux] < 0:
                self.regional_cpu_limits[aux] = inf
        self.providers = gp.tuplelist(self.providers)
        self.prov_regions = gp.tuplelist(self.prov_regions)

        self.prov_regions_vms = []
        self.cpu_vms = {}
        self.gpu_vms = {}
        self.cost_vms = {}
        self.time_aggreg = {}
        self.slowdown = {}
        self.pair_regions = []
        self.comm_slowdown = {}
        self.client_prov_regions_vms = []
        self.time_exec = {}
        self.time_comm = {}

        self.input_data = {}

        self.__read_json(input_file=loader.input_file, job_file=loader.job_file)
        self.pre_process()
        # print(self)
        if not os.path.exists(loader.map_file):
            self.solve()
            # print(self.input_data)
            self.__write_map_json(file_output=loader.map_file)

    def solve(self):
        try:
            # Create optimization model
            model = gp.Model('FL in clouds')

            model.setParam("LogToConsole", 0)

            # Create variables
            x_client_vars = model.addVars(self.client_prov_regions_vms, vtype=GRB.BINARY, name='x')

            y_server_vars = model.addVars(self.prov_regions_vms, vtype=GRB.BINARY, name='y')

            t_m = model.addVar(name="t_m", vtype=GRB.CONTINUOUS, lb=0, ub=inf)

            model.update()

            # Python variables
            vm_costs = \
                gp.quicksum(x_client_vars[i, j, k, l] *
                            self.cost_vms[j, k, l] *
                            t_m for i, j, k, l in self.client_prov_regions_vms) + gp.quicksum(y_server_vars[j, k, l] *
                                                                                              self.cost_vms[j, k, l] *
                                                                                              t_m for j, k, l in
                                                                                              self.prov_regions_vms)

            comm_costs = gp.quicksum(x_client_vars[i, j, k, l] *
                                     y_server_vars[m, n, o] *
                                     ((self.server_msg_train + self.server_msg_test)
                                      * self.cost_transfer[m] +
                                      (self.client_msg_train + self.client_msg_test)
                                      * self.cost_transfer[j]) for i, j, k, l in self.client_prov_regions_vms
                                     for m, n, o in self.prov_regions_vms)

            total_costs = vm_costs + comm_costs

            # normalization
            max_time_exec = 0
            for i, j, k, l in self.time_exec:
                if self.time_exec[i, j, k, l] > max_time_exec:
                    max_time_exec = self.time_exec[i, j, k, l]

            max_time_comm = 0
            for j, k, m, n in self.time_comm:
                if self.time_comm[j, k, m, n] > max_time_comm:
                    max_time_comm = self.time_comm[j, k, m, n]

            max_time_aggreg = 0
            for m, n, o in self.time_aggreg:
                if self.time_aggreg[m, n, o] > max_time_aggreg:
                    max_time_aggreg = self.time_aggreg[m, n, o]

            # print("max_time_exec", max_time_exec)
            # print("max_time_aggreg", max_time_aggreg)
            # print("max_time_comm", max_time_comm)

            max_total_exec = max_time_exec + max_time_aggreg + max_time_comm

            max_vm_cost = 0
            for j, k, l in self.cost_vms:
                if self.cost_vms[j, k, l] > max_vm_cost:
                    max_vm_cost = self.cost_vms[j, k, l]

            max_cost_transfer = 0
            for j in self.cost_transfer:
                for k in self.cost_transfer:
                    comp = (self.server_msg_train + self.server_msg_test) * self.cost_transfer[j] + \
                           (self.client_msg_train + self.client_msg_test) * self.cost_transfer[k]
                    if comp > max_cost_transfer:
                        max_cost_transfer = comp

            max_cost = max_vm_cost * max_total_exec * (len(self.clients) + 1) + max_cost_transfer * \
                (self.server_msg_train + self.server_msg_test + self.client_msg_train +
                 self.client_msg_test) * len(self.clients)

            # print("max_cost", max_cost)
            # print("max_total_exec", max_total_exec)

            # Objective function

            objective_function = self.alpha * (total_costs / max_cost) + (1 - self.alpha) * (t_m / max_total_exec)
            # objective_function = total_costs
            # objective_function = t_m
            model.setObjective(objective_function, GRB.MINIMIZE)

            # Add constraints
            # Budget and deadline constraints
            model.addConstr(total_costs <= self.B_round,
                            "constraint_budget")
            model.addConstr(t_m <= self.T_round,
                            "constraint_deadline")

            # Number of instance types per client and server
            model.addConstrs((x_client_vars.sum(i, '*', '*', '*') == 1 for i in self.clients),
                             "constraint_vm_per_client")

            model.addConstr(y_server_vars.sum('*', '*', '*') == 1,
                            "constraint_vm_server")

            # Global limits
            for p in self.providers:
                model.addConstr(
                    (
                            gp.quicksum(x_client_vars[i, j, k, l] *
                                        self.gpu_vms[j, k, l] for i, j, k, l in self.client_prov_regions_vms if j == p)
                            + gp.quicksum(y_server_vars[j, k, l] *
                                          self.gpu_vms[j, k, l] for j, k, l in self.prov_regions_vms if j == p)
                            <= self.global_gpu_limits[p]
                    ),
                    f"constraint_global_gpu_{p}"
                )

                model.addConstr(
                    (
                            gp.quicksum(x_client_vars[i, j, k, l] *
                                        self.cpu_vms[j, k, l] for i, j, k, l in self.client_prov_regions_vms if j == p)
                            + gp.quicksum(y_server_vars[j, k, l] *
                                          self.cpu_vms[j, k, l] for j, k, l in self.prov_regions_vms if j == p)
                            <= self.global_cpu_limits[p]
                    ),
                    f"constraint_global_cpu_{p}"
                )

            # Regional limits
            for p, r in self.prov_regions:
                model.addConstr(
                    (
                            gp.quicksum(x_client_vars[i, j, k, l] *
                                        self.gpu_vms[j, k, l] for i, j, k, l in self.client_prov_regions_vms if
                                        j == p and k == r)
                            + gp.quicksum(y_server_vars[j, k, l] *
                                          self.gpu_vms[j, k, l] for j, k, l in self.prov_regions_vms if
                                          j == p and k == r)
                            <= self.regional_gpu_limits[p, r]
                    ),
                    f"constraint_regional_gpu_{p}_{r}"
                )

                model.addConstr(
                    (
                            gp.quicksum(x_client_vars[i, j, k, l] *
                                        self.cpu_vms[j, k, l] for i, j, k, l in self.client_prov_regions_vms if
                                        j == p and k == r)
                            + gp.quicksum(y_server_vars[j, k, l] *
                                          self.cpu_vms[j, k, l] for j, k, l in self.prov_regions_vms if
                                          j == p and k == r)
                            <= self.regional_cpu_limits[p, r]
                    ),
                    f"constraint_regional_cpu_{p}_{r}"
                )

            model.addConstrs(
                (
                    t_m >= x_client_vars[i, j, k, l] *
                    y_server_vars[m, n, o] *
                    (
                            self.time_exec[i, j, k, l] +
                            self.time_comm[j, k, m, n] +
                            self.time_aggreg[m, n, o]
                    ) for i, j, k, l in self.client_prov_regions_vms for m, n, o in self.prov_regions_vms
                ),
                "constraint_slowest_client"
            )

            print(model.display())

            model.write("file.lp")

            # Compute optimal solution
            model.optimize()

            # Print solution
            if model.Status == GRB.OPTIMAL:
                obj_value = objective_function.getValue()
                self.input_data['obj_value'] = obj_value
                self.input_data['clients'] = {}
                var_tm = 0
                for v in model.getVars():
                    if v.x == 1:
                        # print("{0} = {1}".format(v.varName, v.x))
                        aux_var_name = v.varName[2:]
                        aux_var_name = aux_var_name[:-1]
                        print(aux_var_name)
                        indexes = aux_var_name.split(',')
                        print(indexes)
                        if v.varName[0] == 'y':
                            # server
                            self.input_data['server'] = {'provider': indexes[0],
                                                         'region': indexes[1],
                                                         'instance_type': indexes[2]}
                        else:
                            # client
                            self.input_data['clients'][indexes[0]] = {'provider': indexes[1],
                                                                      'region': indexes[2],
                                                                      'instance_type': indexes[3]}
                    if v.varName == "t_m":
                        self.input_data['makespan'] = v.x
                        var_tm = v.x

                if self.alpha > 0:
                    cost = ((obj_value - (1 - self.alpha) * (var_tm / max_total_exec)) / self.alpha) * max_cost

                    # print("max_cost", max_cost)
                    # print("max_total_exec", max_total_exec)

                    self.input_data['cost'] = cost
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError as e:
            print('Encountered an attribute error: ' + str(e))

    def __str__(self):
        return f"Values of Mathematical Formulation:\n" \
               f"\tclients: {self.clients}\n" \
               f"\tDS locations: {self.location_ds_clients}\n" \
               f"\tserver_msg_train: {self.server_msg_train}\n" \
               f"\tserver_msg_test: {self.server_msg_test}\n" \
               f"\tclient_msg_train: {self.client_msg_train}\n" \
               f"\tclient_msg_test: {self.client_msg_test}\n" \
               f"\tT_round: {self.T_round}\n" \
               f"\tB_round: {self.B_round}\n" \
               f"\talpha: {self.alpha}\n" \
               f"\n" \
               f"\tproviders: {self.providers}\n" \
               f"\tglobal_cpu_limits: {self.global_cpu_limits}\n" \
               f"\tglobal_gpu_limits: {self.global_gpu_limits}\n" \
               f"\tcost_transfer: {self.cost_transfer}\n" \
               f"\tprov_regions: {self.prov_regions}\n" \
               f"\tregional_cpu_limits: {self.regional_cpu_limits}\n" \
               f"\tregional_gpu_limits: {self.regional_gpu_limits}\n" \
               f"\tprov_regions_vms: {self.prov_regions_vms}\n" \
               f"\tcpu_vms: {self.cpu_vms}\n" \
               f"\tgpu_vms: {self.gpu_vms}\n" \
               f"\tcost_vms: {self.cost_vms}\n" \
               f"\ttime_aggreg: {self.time_aggreg}\n" \
               f"\tslowdown: {self.slowdown}\n" \
               f"\tpair_regions: {self.pair_regions}\n" \
               f"\tcomm_slowdown: {self.comm_slowdown}\n" \
               f"\tbaseline_exec: {self.baseline_exec}\n" \
               f"\tcomm_baseline: {self.comm_baseline}\n" \
               f"\tclient_prov_regions_vms: {self.client_prov_regions_vms}\n" \
               f"\ttime_exec: {self.time_exec}\n" \
               f"\ttime_comm: {self.time_comm}"

    def __repr__(self):
        return self.__str__()

    def __write_map_json(self, file_output):
        logging.info(f"<MathematicalFormulation> Writing {file_output} file")

        with open(file_output, "w") as fp:
            json.dump(self.input_data, fp, sort_keys=True, indent=4, default=str)

    def __read_json(self, input_file, job_file):
        logging.info(f"<MathematicalFormulation> Reading {input_file} file")

        try:
            with open(input_file) as f:
                data = f.read()
            json_data = json.loads(data)
            aux_data = json_data['cpu_vms']
            for provider in aux_data:
                # print(provider)
                for region in aux_data[provider]:
                    # print(region)
                    for vm in aux_data[provider][region]:
                        # print(vm)
                        aux_key = (provider, region, vm)
                        self.prov_regions_vms.append(aux_key)
                        self.cpu_vms[aux_key] = aux_data[provider][region][vm]

            aux_data = json_data['gpu_vms']
            for provider in aux_data:
                # print(provider)
                for region in aux_data[provider]:
                    # print(region)
                    for vm in aux_data[provider][region]:
                        # print(vm)
                        aux_key = (provider, region, vm)
                        test_prov = self.prov_regions_vms.count(aux_key)
                        if test_prov > 0:
                            self.gpu_vms[aux_key] = aux_data[provider][region][vm]
                        else:
                            raise Exception(f"Only GPU values to ({provider}, {region}. {vm})")

            aux_data = json_data['cost_vms']
            for provider in aux_data:
                # print(provider)
                for region in aux_data[provider]:
                    # print(region)
                    for vm in aux_data[provider][region]:
                        # print(vm)
                        aux_key = (provider, region, vm)
                        test_prov = self.prov_regions_vms.count(aux_key)
                        if test_prov > 0:
                            self.cost_vms[aux_key] = aux_data[provider][region][vm]
                        else:
                            raise Exception(f"Only cost to ({provider}, {region}. {vm})")

            aux_data = json_data['time_aggreg']
            for provider in aux_data:
                # print(provider)
                for region in aux_data[provider]:
                    # print(region)
                    for vm in aux_data[provider][region]:
                        # print(vm)
                        aux_key = (provider, region, vm)
                        test_prov = self.prov_regions_vms.count(aux_key)
                        if test_prov > 0:
                            self.time_aggreg[aux_key] = aux_data[provider][region][vm]
                        else:
                            raise Exception(f"Only time to aggregate to ({provider}, {region}. {vm})")

            aux_data = json_data['slowdown']
            for ds_location in aux_data:
                # print(ds_location)
                self.slowdown[ds_location] = {}
                for provider in aux_data[ds_location]:
                    # print(provider)
                    for region in aux_data[ds_location][provider]:
                        # print(region)
                        for vm in aux_data[ds_location][provider][region]:
                            # print(vm)
                            aux_key = (provider, region, vm)
                            test_prov = self.prov_regions_vms.count(aux_key)
                            if test_prov > 0:
                                self.slowdown[ds_location][aux_key] = aux_data[ds_location][provider][region][vm]
                            else:
                                raise Exception(f"Only slowdown time in {ds_location} to ({provider}, {region}. {vm})")

            self.prov_regions_vms = gp.tuplelist(self.prov_regions_vms)

            aux_data = json_data['comm_slowdown']
            for provider_1 in aux_data:
                for region_1 in aux_data[provider_1]:
                    for provider_2 in aux_data[provider_1][region_1]:
                        for region_2 in aux_data[provider_1][region_1][provider_2]:
                            aux = (provider_1, region_1, provider_2, region_2)
                            self.pair_regions.append(aux)
                            self.comm_slowdown[aux] = aux_data[provider_1][region_1][provider_2][region_2]

            self.pair_regions = gp.tuplelist(self.pair_regions)

            if 'baseline_exec' in json_data:
                aux_data = json_data['baseline_exec']
                # TODO: finish later when baseline_exec values are in input.json
            else:
                with open(job_file) as f:
                    data = f.read()
                json_data = json.loads(data)
                aux_data = json_data['tasks']['clients']
                for i in self.clients:
                    self.baseline_exec[i] = aux_data[str(i)]['baseline_exec']

            if 'comm_baseline' in json_data:
                self.comm_baseline = json_data['comm_baseline']
            else:
                with open(job_file) as f:
                    data = f.read()
                json_data = json.loads(data)
                aux_data = json_data['tasks']['server']
                self.comm_baseline = aux_data['comm_baseline']

        except Exception as e:
            logging.error(e)
