from control.util.loader import Loader

import gurobipy as gp
from gurobipy import GRB
from math import inf


class MathematicalFormulationScheduler:

    def pre_process_model_vgg(self):

        self.prov_regions_vms, self.cpu_vms, self.gpu_vms, self.cost_vms, self.time_aggreg,\
        slowdown_us_east_1, slowdown_us_west_2, slowdown_us_central1, slowdown_us_west1 = gp.multidict({
            ('AWS', 'us-east-1', 'g4dn.2xlarge'): [8, 1, 0.752 / 3600, 100000.3, 1.00, 0, 1.00, 0],
            ('AWS', 'us-east-1', 'g3.4xlarge'): [16, 1, 1.14 / 3600, 0.3, 5.09, 0, 1.52, 0],
            ('AWS', 'us-east-1', 't2.xlarge'): [4, 0, 0.1856 / 3600, 0.3, 10, 10, 10, 10],
            ('AWS', 'us-west-2', 'g4dn.2xlarge'): [8, 1, 0.752 / 3600, 0.3, 0.99, 0, 1.36, 0],
            ('AWS', 'us-west-2', 'g3.4xlarge'): [16, 1, 1.14 / 3600, 0.3, 4.44, 0, 1.92, 0],
            ('AWS', 'us-west-2', 't2.xlarge'): [4, 0, 0.1856 / 3600, 0.3, 10, 10, 10, 10],
            ('GCP', 'us-central1', 'n1-standard-8_t4'): [8, 1, 0.730 / 3600, 0.2, 1.03, 0, 0.84, 0],
            ('GCP', 'us-central1', 'n1-standard-16_p4'): [16, 1, 1.360 / 3600, 0.2, 1.28, 0, 0.89, 0],
            ('GCP', 'us-central1', 'n1-standard-8_v100'): [8, 1, 2.860 / 3600, 0.2, 1.04, 0, 0.42, 0],
            ('GCP', 'us-central1', 'e2-standard-4'): [4, 0, 0.134 / 3600, 0.2, 10, 10, 10, 10],
            ('GCP', 'us-west1', 'n1-standard-8_t4'): [8, 1, 0.730 / 3600, 0.2, 1.07, 0, 0.99, 0],
            ('GCP', 'us-west1', 'n1-standard-8_v100'): [8, 1, 2.860 / 3600, 0.2, 1.10, 0, 0.90, 0],
            ('GCP', 'us-west1', 'e2-standard-4'): [4, 0, 0.134 / 3600, 0.2, 10, 10, 10, 10]
        })

        for client in self.clients:
            for prov, region, vm in self.prov_regions_vms:
                if self.gpu_vms[prov, region, vm] == 0:
                    continue
                aux = (client, prov, region, vm)
                # print(aux)
                self.client_prov_regions_vms.append(aux)
                if self.location_ds_clients[client] == 'us-east-1':
                    self.time_exec[aux] = self.baseline_exec[client] * slowdown_us_east_1[prov, region, vm]
                elif self.location_ds_clients[client] == 'us-west-2':
                    self.time_exec[aux] = self.baseline_exec[client] * slowdown_us_west_2[prov, region, vm]
                elif self.location_ds_clients[client] == 'us-central1':
                    self.time_exec[aux] = self.baseline_exec[client] * slowdown_us_central1[prov, region, vm]
                elif self.location_ds_clients[client] == 'us-west1':
                    self.time_exec[aux] = self.baseline_exec[client] * slowdown_us_west1[prov, region, vm]
                else:
                    print(f"We do not support the location of client {client}: {self.location_ds_clients[client]}")
                    return
        self.client_prov_regions_vms = gp.tuplelist(self.client_prov_regions_vms)

        pair_regions, comm_slowdown = gp.multidict({
            ('AWS', 'us-east-1', 'AWS', 'us-east-1'): 1.0,
            ('AWS', 'us-east-1', 'AWS', 'us-west-2'): 5.84,
            ('AWS', 'us-east-1', 'GCP', 'us-central1'): 3.40,
            ('AWS', 'us-east-1', 'GCP', 'us-west1'): 4.78,
            ('AWS', 'us-west-2', 'AWS', 'us-west-2'): 0.97,
            ('AWS', 'us-west-2', 'GCP', 'us-central1'): 4.65,
            ('AWS', 'us-west-2', 'GCP', 'us-west1'): 3.04,
            ('GCP', 'us-central1', 'GCP', 'us-central1'): 0.34,
            ('GCP', 'us-central1', 'GCP', 'us-west1'): 1.09,
            ('GCP', 'us-west1', 'GCP', 'us-west1'): 0.62,
            ('AWS', 'us-west-2', 'AWS', 'us-east-1'): 5.84,
            ('GCP', 'us-central1', 'AWS', 'us-east-1'): 3.40,
            ('GCP', 'us-west1', 'AWS', 'us-east-1'): 4.78,
            ('GCP', 'us-central1', 'AWS', 'us-west-2'): 4.65,
            ('GCP', 'us-west1', 'AWS', 'us-west-2'): 3.04,
            ('GCP', 'us-west1', 'GCP', 'us-central1'): 1.09
        })

        for pair in pair_regions:
            self.time_comm[pair] = self.comm_baseline * comm_slowdown[pair]

    def __init__(self, loader: Loader):
        self.clients = []
        self.location_ds_clients = {}
        for client in loader.job.client_tasks:
            aux = client.client_id
            self.clients.append(aux)
            self.location_ds_clients[aux] = client.bucket_location
        self.clients = gp.tuplelist(self.clients)

        # self.baseline_exec = baseline_exec
        self.baseline_exec = -1
        # self.comm_baseline = comm_baseline
        self.comm_baseline = -1
        self.server_msg_train = loader.job.server_msg_train
        self.server_msg_test = loader.job.server_msg_test
        self.client_msg_train = loader.job.client_msg_train
        self.client_msg_test = loader.job.client_msg_test
        self.T_round = float(loader.mapping_conf.deadline) / loader.job.server_task.n_rounds
        self.B_round = float(loader.mapping_conf.budget) / loader.job.server_task.n_rounds
        self.alpha = loader.mapping_conf.alpha

        self.providers = []
        self.global_cpu_limits = {}
        self.global_gpu_limits = {}
        self.cost_transfer = {}
        self.prov_regions = []
        self.regional_cpu_limits = {}
        self.regional_gpu_limits = {}

        for region_id, region in loader.loc.items():
            aux = region.provider.upper()
            test_prov = self.providers.count(aux)
            if test_prov > 0:
                if self.global_cpu_limits[aux] != region.global_cpu_limit:
                    raise Exception(f"Global vCPUs limit different in region {region_id}")
                if self.global_gpu_limits[aux] != region.global_gpu_limit:
                    raise Exception(f"Global GPUs limit different in region {region_id}")
                if self.cost_transfer[aux] != region.cost_transfer:
                    raise Exception(f"Cost transfer different in region {region_id}")
            else:
                self.providers.append(aux)
                self.global_cpu_limits[aux] = region.global_cpu_limit
                self.global_gpu_limits[aux] = region.global_gpu_limit
                self.cost_transfer[aux] = region.cost_transfer
            aux = (aux, region.region)
            self.prov_regions.append(aux)
            self.regional_gpu_limits[aux] = region.regional_gpu_limit
            self.regional_cpu_limits[aux] = region.regional_cpu_limit
        self.providers = gp.tuplelist(self.providers)
        self.prov_regions = gp.tuplelist(self.prov_regions)

        self.prov_regions_vms = None
        self.cpu_vms = None
        self.gpu_vms = None
        self.cost_vms = None
        self.time_aggreg = None
        self.client_prov_regions_vms = []
        self.time_exec = {}
        self.time_comm = {}

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

            # Compute optimal solution
            model.optimize()

            # print(model.display())

            # model.write("file.lp")

            # Print solution
            if model.Status == GRB.OPTIMAL:
                obj_value = objective_function.getValue()
                print("Objective Function Value = {0}".format(obj_value))
                var_tm = 0
                for v in model.getVars():
                    if v.x == 1:
                        print("{0} = {1}".format(v.varName, v.x))
                    if v.varName == "t_m":
                        print("{0} = {1}".format(v.varName, v.x))
                        var_tm = v.x

                if self.alpha > 0:
                    cost = ((obj_value - (1 - self.alpha) * (var_tm / max_total_exec)) / self.alpha) * max_cost

                    # print("max_cost", max_cost)
                    # print("max_total_exec", max_total_exec)

                    print("Computed cost = ", cost)
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError as e:
            print('Encountered an attribute error: ' + str(e))

    def __str__(self):
        return f"Values of Mathematical Formulation:\n" \
               f"\tclients: {self.clients}\n" \
               f"\tDS locations: {self.location_ds_clients}\n" \
               f"\tserver_msg_train: {self.server_msg_train}\n" \
               f"\tserver_msg_test: {self.server_msg_test}\n"
        self.server_msg_test = loader.job.server_msg_test
        self.client_msg_train = loader.job.client_msg_train
        self.client_msg_test = loader.job.client_msg_test
        self.T_round = float(loader.mapping_conf.deadline) / loader.job.server_task.n_rounds
        self.B_round = float(loader.mapping_conf.budget) / loader.job.server_task.n_rounds
        self.alpha = loader.mapping_conf.alpha

        self.providers = []
        self.global_cpu_limits = {}
        self.global_gpu_limits = {}
        self.cost_transfer = {}
        self.prov_regions = []
        self.regional_cpu_limits = {}
        self.regional_gpu_limits = {}
        self.prov_regions_vms = None
        self.cpu_vms = None
        self.gpu_vms = None
        self.cost_vms = None
        self.time_aggreg = None
        self.client_prov_regions_vms = []
        self.time_exec = {}
        self.time_comm = {}

    def __repr__(self):
        return self.__str__()
