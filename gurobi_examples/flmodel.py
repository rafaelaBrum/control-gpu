# First version of FL optimization problem in clouds

import gurobipy as gp
from gurobipy import GRB
from math import inf


def solve(client_prov_regions_vms, cost_transfer, prov_regions_vms, cost_vms, server_msg_train, server_msg_test,
          client_msg_train, client_msg_test, T_round, clients, B_round, alpha_1, alpha_2,
          providers, gpu_vms, global_gpu_limits, cpu_vms, global_cpu_limits, regional_gpu_limits, prov_regions,
          regional_cpu_limits, time_exec, time_comm, time_aggreg):
    try:
        # Create optimization model
        model = gp.Model('FL in clouds')

        # model.setParam("LogToConsole", 0)

        # Create variables
        x_client_vars = model.addVars(client_prov_regions_vms, vtype=GRB.BINARY, name='x')

        y_server_vars = model.addVars(prov_regions_vms, vtype=GRB.BINARY, name='y')

        t_m = model.addVar(name="t_m", vtype=GRB.CONTINUOUS, lb=0, ub=inf)

        model.update()

        # Python variables
        # vm_costs = gp.quicksum(x_client_vars[i, j, k, l] *
        #                        cost_vms[j, k, l] *
        #                        t_m for i in clients
        #                        for j, k, l in vms_region_prov) + \
        #            gp.quicksum(y_server_vars[j, k, l] *
        #                        cost_vms[j, k, l] *
        #                        t_m for j, k, l in vms_region_prov)
        vm_costs = \
            gp.quicksum(x_client_vars[i, j, k, l] *
                        cost_vms[j, k, l] *
                        t_m for i, j, k, l in client_prov_regions_vms) + gp.quicksum(y_server_vars[j, k, l] *
                                                                                     cost_vms[j, k, l] *
                                                                                     t_m for j, k, l in prov_regions_vms)

        # comm_costs = gp.quicksum(x_client_vars[i, j, k, l] *
        #                          y_server_vars[m, n, o] *
        #                          ((server_msg_train + server_msg_test)
        #                           * cost_transfer[j] +
        #                           (client_msg_train + client_msg_test)
        #                           * cost_transfer[m]) for i in clients
        #                          for j, k, l in vms_region_prov
        #                          for m, n, o in vms_region_prov)
        comm_costs = gp.quicksum(x_client_vars[i, j, k, l] *
                                 y_server_vars[m, n, o] *
                                 ((server_msg_train + server_msg_test)
                                  * cost_transfer[j] +
                                  (client_msg_train + client_msg_test)
                                  * cost_transfer[m]) for i, j, k, l in client_prov_regions_vms
                                 for m, n, o in prov_regions_vms)

        total_costs = vm_costs + comm_costs

        max_vm_cost = 0

        for j, k, l in cost_vms:
            if cost_vms[j, k, l] > max_vm_cost:
                max_vm_cost = cost_vms[j, k, l]

        max_cost_transfer = 0
        for j in cost_transfer:
            if cost_transfer[j] > max_cost_transfer:
                max_cost_transfer = cost_transfer[j]

        max_cost = max_vm_cost * T_round * (len(clients) + 1) + max_cost_transfer * (server_msg_train +
                                                                                     server_msg_test +
                                                                                     client_msg_train +
                                                                                     client_msg_test) * len(clients)

        # Objective function

        objective_function = alpha_1 * (total_costs / max_cost) + alpha_2 * (t_m / T_round)
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
                "constraint_global_gpu"
            )

            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    cpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      cpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p)
                        <= global_cpu_limits[p]
                ),
                "constraint_global_cpu"
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
                "constraint_regional_gpu"
            )

            model.addConstr(
                (
                        gp.quicksum(x_client_vars[i, j, k, l] *
                                    cpu_vms[j, k, l] for i, j, k, l in client_prov_regions_vms if j == p and k == r)
                        + gp.quicksum(y_server_vars[j, k, l] *
                                      cpu_vms[j, k, l] for j, k, l in prov_regions_vms if j == p and k == r)
                        <= regional_cpu_limits[p, r]
                ),
                "constraint_regional_cpu"
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

        # Print solution
        if model.Status == GRB.OPTIMAL:
            print("Objective Function Value = {0}".format(objective_function.getValue()))
            for v in model.getVars():
                if v.x == 1 or v.varName == "t_m":
                    print("{0} = {1}".format(v.varName, v.x))
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError as e:
        print('Encountered an attribute error: ' + str(e))
