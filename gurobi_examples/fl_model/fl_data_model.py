import flmodel
import gurobipy as gp
from math import inf

from datetime import datetime


def pre_process_model_vgg(clients, location, baseline_exec, comm_baseline, server_msg_train, server_msg_test,
                          client_msg_train, client_msg_test, T_round, B_round, alpha):

    providers, global_cpu_limits, global_gpu_limits, cost_transfer = gp.multidict({
        'AWS': [inf, inf, 0.090],
        'GCP': [40, 4, 0.120]
    })
    # providers, global_cpu_limits, global_gpu_limits, cost_transfer = gp.multidict({
    #     'AWS': [inf, inf, 0.090],
    #     'GCP': [inf, inf, 0.120]
    # })

    prov_regions, regional_cpu_limits, regional_gpu_limits = gp.multidict({
        ('AWS', 'us-east-1'): [52, inf], # 48 vCPUs of G family and 4 for T family
        ('AWS', 'us-west-2'): [36, inf], # 32 vCPUs of G family and 4 for T family
        ('GCP', 'us-central1'): [40, 4],
        ('GCP', 'us-west1'): [40, 4]
    })
    # prov_regions, regional_cpu_limits, regional_gpu_limits = gp.multidict({
    #     ('AWS', 'us-east-1'): [inf, inf],  # 48 vCPUs of G family and 4 for T family
    #     ('AWS', 'us-west-2'): [inf, inf],  # 32 vCPUs of G family and 4 for T family
    #     ('GCP', 'us-central1'): [inf, inf],
    #     ('GCP', 'us-west1'): [inf, inf]
    # })

    prov_regions_vms, cpu_vms, gpu_vms, cost_vms, \
    time_aggreg, slowdown_us_east_1, slowdown_us_west_2, slowdown_us_central1, slowdown_us_west1 = gp.multidict({
        ('AWS', 'us-east-1', 'g4dn.2xlarge'):           [8, 1,  0.752/3600, 100000.3, 1.00, 0, 1.00, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge'):            [16, 1,  1.14/3600, 0.3, 5.09, 0, 1.52, 0],
        # ('AWS', 'us-east-1', 'p3.2xlarge'):          [16, 1,  3.06/3600, 0.3, 4.82, 0, 0, 0],
        ('AWS', 'us-east-1', 't2.xlarge'):              [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 'g4dn.2xlarge'):           [8, 1,  0.752/3600, 0.3, 0.99, 0, 1.36, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge'):            [16, 1,  1.14/3600, 0.3, 4.44, 0, 1.92, 0],
        ('AWS', 'us-west-2', 't2.xlarge'):              [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'n1-standard-8_t4'):     [8, 1,  0.730/3600, 0.2, 1.03, 0, 0.84, 0],
        ('GCP', 'us-central1', 'n1-standard-16_p4'):   [16, 1,  1.360/3600, 0.2, 1.28, 0, 0.89, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100'):   [8, 1,  2.860/3600, 0.2, 1.04, 0, 0.41, 0],
        ('GCP', 'us-central1', 'e2-standard-4'):        [4, 0,  0.134/3600, 0.2, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'n1-standard-8_t4'):        [8, 1,  0.730/3600, 0.2, 1.07, 0, 0.99, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100'):      [8, 1,  2.860/3600, 0.2, 1.10, 0, 0.90, 0],
        ('GCP', 'us-west1', 'e2-standard-4'):           [4, 0,  0.134/3600, 0.2, 10, 10, 10, 10]
    })

    client_prov_regions_vms = []
    time_exec = {}
    for client in clients:
        for prov, region, vm in prov_regions_vms:
            if gpu_vms[prov, region, vm] == 0:
                continue
            # if client % 2 == 0 and vm in "n1-standard-8_t4":
            #     continue
            # if client % 2 != 0 and vm in "n1-standard-8_v100":
            #     continue
            # if location[client] == 'us-east-1' and vm in "n1-standard-8_t4":
            #     continue
            # if location[client] == 'us-central1' and vm in "g4dn.2xlarge":
            #     continue
            aux = (client, prov, region, vm)
            # print(aux)
            client_prov_regions_vms.append(aux)
            if location[client] == 'us-east-1':
                time_exec[aux] = baseline_exec[client]*slowdown_us_east_1[prov, region, vm]
                # if vm in 'g4dn.2xlarge':
                # print(f"time_exec{aux}]", time_exec[aux])
            elif location[client] == 'us-west-2':
                time_exec[aux] = baseline_exec[client] * slowdown_us_west_2[prov, region, vm]
            elif location[client] == 'us-central1':
                time_exec[aux] = baseline_exec[client] * slowdown_us_central1[prov, region, vm]
                # print(f"baseline_exec[{client}]", baseline_exec[client])
                # print(f"slowdown_us_central1[{prov}, {region}, {vm}]", slowdown_us_central1[prov, region, vm])
                # print(f"time_exec{aux}]", time_exec[aux])
            elif location[client] == 'us-west1':
                time_exec[aux] = baseline_exec[client] * slowdown_us_west1[prov, region, vm]
            else:
                print(f"We do not support the location of client {client}: {location[client]}")
                exit()
    client_prov_regions_vms = gp.tuplelist(client_prov_regions_vms)

    # print("tempo_aggreg")
    # print(time_aggreg)

    # print("client_prov_regions_vms", client_prov_regions_vms)
    # print("time_exec", time_exec)

    # print("cost_vms")
    # print(cost_vms)
    #
    # print("cost_transfer")
    # print(cost_transfer)
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

    time_comm = {}

    for pair in pair_regions:
        time_comm[pair] = comm_baseline*comm_slowdown[pair]
        # print(f"time_comm[{pair}] = {time_comm[pair]}")

    start_timestamp = datetime.now()

    flmodel.solve(client_prov_regions_vms=client_prov_regions_vms, cost_transfer=cost_transfer,
                  prov_regions_vms=prov_regions_vms, cost_vms=cost_vms, server_msg_train=server_msg_train,
                  server_msg_test=server_msg_test, client_msg_train=client_msg_train, client_msg_test=client_msg_test,
                  T_round=T_round, clients=clients, B_round=B_round, alpha=alpha, providers=providers,
                  gpu_vms=gpu_vms, global_gpu_limits=global_gpu_limits, cpu_vms=cpu_vms,
                  global_cpu_limits=global_cpu_limits, regional_gpu_limits=regional_gpu_limits,
                  prov_regions=prov_regions, regional_cpu_limits=regional_cpu_limits, time_exec=time_exec,
                  time_comm=time_comm, time_aggreg=time_aggreg)

    end_timestamp = datetime.now()
    return end_timestamp - start_timestamp
