# First version of FL optimization problem in clouds

import gurobipy as gp
from gurobipy import GRB
from math import inf


def solve(client_prov_regions_vms, cost_transfer, prov_regions_vms, cost_vms, server_msg_train, server_msg_test,
          client_msg_train, client_msg_test, T_round, clients, B_round, alpha, providers, gpu_vms, global_gpu_limits,
          cpu_vms, global_cpu_limits, regional_gpu_limits, prov_regions, regional_cpu_limits, time_exec, time_comm,
          time_aggreg):
    try:
        # Create optimization model
        model = gp.Model('FL in clouds')

        model.setParam("LogToConsole", 0)

        # Create variables
        x_client_vars = model.addVars(client_prov_regions_vms, vtype=GRB.BINARY, name='x')

        y_server_vars = model.addVars(prov_regions_vms, vtype=GRB.BINARY, name='y')

        t_m = model.addVar(name="t_m", vtype=GRB.CONTINUOUS, lb=0, ub=inf)

        model.update()

        # Python variables
        vm_costs = \
            gp.quicksum(x_client_vars[i, j, k, l] *
                        cost_vms[j, k, l] *
                        t_m for i, j, k, l in client_prov_regions_vms) + gp.quicksum(y_server_vars[j, k, l] *
                                                                                     cost_vms[j, k, l] *
                                                                                     t_m for j, k, l in prov_regions_vms)

        comm_costs = gp.quicksum(x_client_vars[i, j, k, l] *
                                 y_server_vars[m, n, o] *
                                 ((server_msg_train + server_msg_test)
                                  * cost_transfer[m] +
                                  (client_msg_train + client_msg_test)
                                  * cost_transfer[j]) for i, j, k, l in client_prov_regions_vms
                                 for m, n, o in prov_regions_vms)

        total_costs = vm_costs + comm_costs

        # normalization
        max_time_exec = 0
        for i, j, k, l in time_exec:
            if time_exec[i, j, k, l] > max_time_exec:
                # print(f"time_exec ({i}, {j}, {k}, {l})", time_exec[i, j, k, l])
                max_time_exec = time_exec[i, j, k, l]

        max_time_comm = 0
        for j, k, m, n in time_comm:
            if time_comm[j, k, m, n] > max_time_comm:
                # print(f"time_comm ({j}, {k}, {m}, {n})", time_comm[j, k, m, n])
                max_time_comm = time_comm[j, k, m, n]

        max_time_aggreg = 0
        for m, n, o in time_aggreg:
            if time_aggreg[m, n, o] > max_time_aggreg:
                # print(f"time_aggreg ({m}, {n}, {o})", time_aggreg[m, n, o])
                max_time_aggreg = time_aggreg[m, n, o]

        # print("max_time_exec", max_time_exec)
        # print("max_time_aggreg", max_time_aggreg)
        # print("max_time_comm", max_time_comm)

        max_total_exec = max_time_exec + max_time_aggreg + max_time_comm

        max_vm_cost = 0
        for j, k, l in cost_vms:
            if cost_vms[j, k, l] > max_vm_cost:
                # print(f"max_vm_cost ({j}, {k}, {l})", cost_vms[j, k, l])
                max_vm_cost = cost_vms[j, k, l]

        max_cost_transfer = 0
        for j in cost_transfer:
            if cost_transfer[j] > max_cost_transfer:
                # print(f"max_cost_transfer ({j})", cost_transfer[j])
                max_cost_transfer = cost_transfer[j]

        max_cost = max_vm_cost * max_total_exec * (len(clients) + 1) \
                   + max_cost_transfer * (server_msg_train +
                                          server_msg_test +
                                          client_msg_train +
                                          client_msg_test) * len(clients)

        # print("max_cost", max_cost)
        # print("max_total_exec", max_total_exec)

        # Objective function

        objective_function = alpha * (total_costs / max_cost) + (1-alpha) * (t_m / max_total_exec)
        model.setObjective(objective_function, GRB.MINIMIZE)

        # Add constraints
        # Budget and deadline constraints
        model.addConstr(total_costs <= B_round,
                        "constraint_budget")
        model.addConstr(t_m <= T_round,
                        "constraint_deadline")

        # Number of instance types per client and server
        model.addConstrs((x_client_vars.sum(i, '*', '*', '*') == 1 for i in clients),
                         "constraint_vm_per_client")

        model.addConstr(y_server_vars.sum('*', '*', '*') == 1,
                        "constraint_vm_server")

        # Global limits
        for p in providers:
            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    gpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      gpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p)
                        <= global_gpu_limits[p]
                ),
                f"constraint_global_gpu_{p}"
            )

            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    cpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      cpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p)
                        <= global_cpu_limits[p]
                ),
                f"constraint_global_cpu_{p}"
            )

        # Regional limits
        for p, r in prov_regions:
            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    gpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p and k == r)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      gpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p and k == r)
                        <= regional_gpu_limits[p, r]
                ),
                f"constraint_regional_gpu_{p}_{r}"
            )

            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    cpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p and k == r)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      cpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p and k == r)
                        <= regional_cpu_limits[p, r]
                ),
                f"constraint_regional_cpu_{p}_{r}"
            )

        model.addConstrs(
            (
                    x_client_vars[i, j, k, l] *
                    y_server_vars[m, n, o] *
                    (
                            time_exec[i, j, k, l] +
                            time_comm[j, k, m, n] +
                            time_aggreg[m, n, o]
                    ) <= t_m for i, j, k, l in client_prov_regions_vms for m, n, o in prov_regions_vms
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

            if alpha > 0:
                cost = ((obj_value*1/alpha) - (var_tm / max_total_exec))*max_cost

            # print("max_cost", max_cost)
            # print("max_total_exec", max_total_exec)

                print("Computed cost = ", cost)
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError as e:
        print('Encountered an attribute error: ' + str(e))
