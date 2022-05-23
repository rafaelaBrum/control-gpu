import flmodel
import gurobipy as gp
from math import inf

from datetime import datetime

#TODO: change slowdown of n1-standard-8_t4 in us-west1 !!!!


def pre_process_model_vgg(clients, location, baseline_exec, comm_baseline, server_msg_train, server_msg_test,
                          client_msg_train, client_msg_test, T_round, B_round, alpha):
    providers, global_cpu_limits, global_gpu_limits, cost_transfer = gp.multidict({
        'AWS': [inf, inf, 0.090],
        'GCP': [inf, inf, 0.120]
    })

    prov_regions, regional_cpu_limits, regional_gpu_limits = gp.multidict({
        ('AWS', 'us-east-1'): [inf, inf],
        ('AWS', 'us-west-2'): [inf, inf],
        ('GCP', 'us-central1'): [inf, inf],
        ('GCP', 'us-west1'): [inf, inf]
    })


    prov_regions_vms, cpu_vms, gpu_vms, cost_vms, \
    time_aggreg, slowdown_us_east_1, slowdown_us_west_2, slowdown_us_central1, slowdown_us_west1 = gp.multidict({
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Real'): [8, 1,  0.752/3600, 0.3, 1.00, 0, 0, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake1'): [8, 1, 0.639/3600, 0.36, 1.20, 0, 0, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake2'): [8, 1, 0.526/3600, 0.45, 1.50, 0, 0, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake3'): [8, 1, 0.602/3600, 0.36, 1.20, 0, 0, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake4'): [8, 1, 0.902/3600, 0.255, 0.85, 0, 0, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake5'): [8, 1, 1.128/3600, 0.21, 0.70, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Real'): [16, 1,  1.14/3600, 0.3, 4.98, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake1'): [16, 1, 0.969/3600, 0.36, 5.98, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake2'): [16, 1, 0.798/3600, 0.45, 7.48, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake3'): [16, 1, 0.912/3600, 0.36, 5.98, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake4'): [16, 1, 1.368/3600, 0.255, 4.24, 0, 0, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake5'): [16, 1, 1.710/3600, 0.21, 3.49, 0, 0, 0],
        ('AWS', 'us-east-1', 't2.xlarge_Real'): [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake1'): [4, 0, 0.1578/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake2'): [4, 0, 0.1299/3600, 0.45, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake3'): [4, 0, 0.1485/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake4'): [4, 0, 0.2227/3600, 0.255, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake5'): [4, 0, 0.2784/3600, 0.21, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Real'): [8, 1,  0.752/3600, 0.3, 0.85, 0, 0, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake1'): [8, 1, 0.639/3600, 0.36, 1.02, 0, 0, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake2'): [8, 1, 0.526/3600, 0.45, 1.28, 0, 0, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake3'): [8, 1, 0.602/3600, 0.36, 1.02, 0, 0, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake4'): [8, 1, 0.902/3600, 0.255, 0.72, 0, 0, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake5'): [8, 1, 1.128/3600, 0.21, 0.60, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Real'): [16, 1,  1.140/3600, 0.3, 4.35, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake1'): [16, 1, 0.969/3600, 0.36, 5.22, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake2'): [16, 1, 0.798/3600, 0.45, 6.52, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake3'): [16, 1, 0.912/3600, 0.36, 5.22, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake4'): [16, 1, 1.368/3600, 0.255, 3.70, 0, 0, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake5'): [16, 1, 1.710/3600, 0.21, 3.04, 0, 0, 0],
        ('AWS', 'us-west-2', 't2.xlarge_Real'): [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake1'): [4, 0, 0.1578/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake2'): [4, 0, 0.1299/3600, 0.45, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake3'): [4, 0, 0.1485/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake4'): [4, 0, 0.2227/3600, 0.255, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake5'): [4, 0, 0.2784/3600, 0.21, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Real'): [8, 1,  0.730/3600, 0.2, 1.01, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake1'): [8, 1, 0.620/3600, 0.24, 1.21, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake2'): [8, 1, 0.511/3600, 0.3, 1.51, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake3'): [8, 1, 0.584/3600, 0.24, 1.21, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake4'): [8, 1, 0.876/3600, 0.17, 0.86, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake5'): [8, 1, 1.095/3600, 0.14, 0.71, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Real'): [8, 1,  1.360/3600, 0.2, 1.25, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake1'): [8, 1, 1.156/3600, 0.24, 1.50, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake2'): [8, 1, 0.952/3600, 0.3, 1.87, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake3'): [8, 1, 1.088/3600, 0.24, 1.50, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake4'): [8, 1, 1.632/3600, 0.17, 1.06, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake5'): [8, 1, 2.040/3600, 0.14, 0.87, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Real'): [8, 1,  2.860/3600, 0.2, 1.01, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake1'): [8, 1, 2.431/3600, 0.24, 1.22, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake2'): [8, 1, 2.002/3600, 0.3, 1.52, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake3'): [8, 1, 2.288/3600, 0.24, 1.22, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake4'): [8, 1, 3.432/3600, 0.17, 0.86, 0, 0, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake5'): [8, 1, 4.290/3600, 0.2, 0.71, 0, 0, 0],
        ('GCP', 'us-central1', 'e2-standard-4_Real'): [4, 0,  0.134/3600, 0.2, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake1'): [4, 0, 0.114/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake2'): [4, 0, 0.094/3600, 0.3, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake3'): [4, 0, 0.107/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake4'): [4, 0, 0.161/3600, 0.17, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake5'): [4, 0, 0.201/3600, 0.14, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Real'): [8, 1,  0.730/3600, 0.2, 1.04, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake1'): [8, 1, 0.620/3600, 0.24, 1.25, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake2'): [8, 1, 0.511/3600, 0.3, 1.56, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake3'): [8, 1, 0.584/3600, 0.24, 1.25, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake4'): [8, 1, 0.876/3600, 0.17, 0.89, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake5'): [8, 1, 1.095/3600, 0.14, 0.73, 0.98, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Real'): [8, 1,  2.860/3600, 0.2, 1.08, 0, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake1'): [8, 1, 2.431/3600, 0.24, 1.29, 0, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake2'): [8, 1, 2.002/3600, 0.3, 1.61, 0, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake3'): [8, 1, 2.228/3600, 0.24, 1.29, 0, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake4'): [8, 1, 3.432/3600, 0.17, 0.91, 0, 0, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake5'): [8, 1, 4.290/3600, 0.14, 0.75, 0, 0, 0],
        ('GCP', 'us-west1', 'e2-standard-4_Real'): [4, 0,  0.134/3600, 0.2, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'e2-standard-4_Fake1'): [4, 0, 0.114/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'e2-standard-4_Fake2'): [4, 0, 0.094/3600, 0.3, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'e2-standard-4_Fake3'): [4, 0, 0.107/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'e2-standard-4_Fake4'): [4, 0, 0.161/3600, 0.17, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'e2-standard-4_Fake5'): [4, 0, 0.201/3600, 0.14, 10, 10, 10, 10]
    })
    client_prov_regions_vms = []
    time_exec = {}
    for client in clients:
        for prov, region, vm in prov_regions_vms:
            if gpu_vms[prov, region, vm] == 0:
                continue
            aux = (client, prov, region, vm)
            # print(aux)
            client_prov_regions_vms.append(aux)
            if location[client] == 'us-east-1':
                time_exec[aux] = baseline_exec[client]*slowdown_us_east_1[prov, region, vm]
                # if vm in 'g4dn.2xlarge':
                #     print(f"time_exec{aux}]", time_exec[aux])
            elif location[client] == 'us-west-2':
                time_exec[aux] = baseline_exec[client] * slowdown_us_west_2[prov, region, vm]
            elif location[client] == 'us-central1':
                time_exec[aux] = baseline_exec[client] * slowdown_us_central1[prov, region, vm]
            elif location[client] == 'us-west1':
                time_exec[aux] = baseline_exec[client] * slowdown_us_west1[prov, region, vm]
            else:
                print(f"We do not support the location of client {client}: {location[client]}")
                exit()
    client_prov_regions_vms = gp.tuplelist(client_prov_regions_vms)

    # print("client_prov_regions_vms", client_prov_regions_vms)
    # print("time_exec", time_exec)
    #
    # exit()

    pair_regions, comm_slowdown = gp.multidict({
        ('AWS', 'us-east-1', 'AWS', 'us-east-1'): 1.0,
        ('AWS', 'us-east-1', 'AWS', 'us-west-2'): 6.74,
        ('AWS', 'us-east-1', 'GCP', 'us-central1'): 4.97,
        ('AWS', 'us-east-1', 'GCP', 'us-west1'): 5.41,
        ('AWS', 'us-west-2', 'AWS', 'us-west-2'): 1.12,
        ('AWS', 'us-west-2', 'GCP', 'us-central1'): 5.74,
        ('AWS', 'us-west-2', 'GCP', 'us-west1'): 3.51,
        ('GCP', 'us-central1', 'GCP', 'us-central1'): 0.39,
        ('GCP', 'us-central1', 'GCP', 'us-west1'): 1.26,
        ('GCP', 'us-west1', 'GCP', 'us-west1'): 0.72,
        ('AWS', 'us-west-2', 'AWS', 'us-east-1'): 6.74,
        ('GCP', 'us-central1', 'AWS', 'us-east-1'): 4.97,
        ('GCP', 'us-west1', 'AWS', 'us-east-1'): 5.41,
        ('GCP', 'us-central1', 'AWS', 'us-west-2'): 5.74,
        ('GCP', 'us-west1', 'AWS', 'us-west-2'): 3.51,
        ('GCP', 'us-west1', 'GCP', 'us-central1'): 1.26
    })

    time_comm = {}

    for pair in pair_regions:
        time_comm[pair] = comm_baseline*comm_slowdown[pair]
        # print(f"time_comm´[{}]")

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
