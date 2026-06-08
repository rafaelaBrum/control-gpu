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
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Real'): [8, 1,  0.752/3600, 0.3, 1.00, 0, 1.00, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake1'): [8, 1, 0.639/3600, 0.36, 1.20, 0, 1.20, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake2'): [8, 1, 0.526/3600, 0.45, 1.50, 0, 1.50, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake3'): [8, 1, 0.602/3600, 0.36, 1.20, 0, 1.20, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake4'): [8, 1, 0.902/3600, 0.255, 0.85, 0, 0.85, 0],
        ('AWS', 'us-east-1', 'g4dn.2xlarge_Fake5'): [8, 1, 1.128/3600, 0.21, 0.70, 0, 0.70, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Real'): [16, 1,  1.14/3600, 0.3, 4.92, 0, 1.52, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake1'): [16, 1, 0.969/3600, 0.36, 5.91, 0, 1.82, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake2'): [16, 1, 0.798/3600, 0.45, 7.38, 0, 2.28, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake3'): [16, 1, 0.912/3600, 0.36, 5.91, 0, 1.82, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake4'): [16, 1, 1.368/3600, 0.255, 4.18, 0, 1.29, 0],
        ('AWS', 'us-east-1', 'g3.4xlarge_Fake5'): [16, 1, 1.710/3600, 0.21, 3.44, 0, 1.06, 0],
        ('AWS', 'us-east-1', 't2.xlarge_Real'): [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake1'): [4, 0, 0.1578/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake2'): [4, 0, 0.1299/3600, 0.45, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake3'): [4, 0, 0.1485/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake4'): [4, 0, 0.2227/3600, 0.255, 10, 10, 10, 10],
        ('AWS', 'us-east-1', 't2.xlarge_Fake5'): [4, 0, 0.2784/3600, 0.21, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Real'): [8, 1,  0.752/3600, 0.3, 0.99, 0, 1.36, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake1'): [8, 1, 0.639/3600, 0.36, 1.18, 0, 1.63, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake2'): [8, 1, 0.526/3600, 0.45, 1.48, 0, 2.03, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake3'): [8, 1, 0.602/3600, 0.36, 1.12, 0, 1.63, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake4'): [8, 1, 0.902/3600, 0.255, 0.84, 0, 1.15, 0],
        ('AWS', 'us-west-2', 'g4dn.2xlarge_Fake5'): [8, 1, 1.128/3600, 0.21, 0.69, 0, 0.95, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Real'): [16, 1,  1.140/3600, 0.3, 4.44, 0, 1.92, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake1'): [16, 1, 0.969/3600, 0.36, 5.33, 0, 2.31, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake2'): [16, 1, 0.798/3600, 0.45, 6.66, 0, 2.88, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake3'): [16, 1, 0.912/3600, 0.36, 5.33, 0, 2.31, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake4'): [16, 1, 1.368/3600, 0.255, 3.77, 0, 1.63, 0],
        ('AWS', 'us-west-2', 'g3.4xlarge_Fake5'): [16, 1, 1.710/3600, 0.21, 3.11, 0, 1.35, 0],
        ('AWS', 'us-west-2', 't2.xlarge_Real'): [4, 0,  0.1856/3600, 0.3, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake1'): [4, 0, 0.1578/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake2'): [4, 0, 0.1299/3600, 0.45, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake3'): [4, 0, 0.1485/3600, 0.36, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake4'): [4, 0, 0.2227/3600, 0.255, 10, 10, 10, 10],
        ('AWS', 'us-west-2', 't2.xlarge_Fake5'): [4, 0, 0.2784/3600, 0.21, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Real'): [8, 1,  0.730/3600, 0.2, 1.03, 0, 0.85, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake1'): [8, 1, 0.620/3600, 0.24, 1.23, 0, 1.01, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake2'): [8, 1, 0.511/3600, 0.3, 1.54, 0, 1.27, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake3'): [8, 1, 0.584/3600, 0.24, 1.23, 0, 1.01, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake4'): [8, 1, 0.876/3600, 0.17, 0.87, 0, 0.72, 0],
        ('GCP', 'us-central1', 'n1-standard-8_t4_Fake5'): [8, 1, 1.095/3600, 0.14, 0.72, 0, 0.59, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Real'): [8, 1,  1.360/3600, 0.2, 1.28, 0, 0.89, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake1'): [8, 1, 1.156/3600, 0.24, 1.53, 0, 1.07, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake2'): [8, 1, 0.952/3600, 0.3, 1.91, 0, 1.34, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake3'): [8, 1, 1.088/3600, 0.24, 1.53, 0, 1.07, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake4'): [8, 1, 1.632/3600, 0.17, 1.08, 0, 0.76, 0],
        ('GCP', 'us-central1', 'n1-standard-8_p4_Fake5'): [8, 1, 2.040/3600, 0.14, 0.89, 0, 0.62, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Real'): [8, 1,  2.860/3600, 0.2, 1.04, 0, 0.41, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake1'): [8, 1, 2.431/3600, 0.24, 1.24, 0, 0.49, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake2'): [8, 1, 2.002/3600, 0.3, 1.55, 0, 0.61, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake3'): [8, 1, 2.288/3600, 0.24, 1.24, 0, 0.49, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake4'): [8, 1, 3.432/3600, 0.17, 0.88, 0, 0.35, 0],
        ('GCP', 'us-central1', 'n1-standard-8_v100_Fake5'): [8, 1, 4.290/3600, 0.2, 0.72, 0, 0.29, 0],
        ('GCP', 'us-central1', 'e2-standard-4_Real'): [4, 0,  0.134/3600, 0.2, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake1'): [4, 0, 0.114/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake2'): [4, 0, 0.094/3600, 0.3, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake3'): [4, 0, 0.107/3600, 0.24, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake4'): [4, 0, 0.161/3600, 0.17, 10, 10, 10, 10],
        ('GCP', 'us-central1', 'e2-standard-4_Fake5'): [4, 0, 0.201/3600, 0.14, 10, 10, 10, 10],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Real'): [8, 1,  0.730/3600, 0.2, 1.06, 0, 0.99, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake1'): [8, 1, 0.620/3600, 0.24, 1.28, 0, 1.18, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake2'): [8, 1, 0.511/3600, 0.3, 1.60, 0, 1.48, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake3'): [8, 1, 0.584/3600, 0.24, 1.28, 0, 1.18, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake4'): [8, 1, 0.876/3600, 0.17, 0.90, 0, 0.84, 0],
        ('GCP', 'us-west1', 'n1-standard-8_t4_Fake5'): [8, 1, 1.095/3600, 0.14, 0.74, 0, 0.69, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Real'): [8, 1,  2.860/3600, 0.2, 1.10, 0, 0.90, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake1'): [8, 1, 2.431/3600, 0.24, 1.32, 0, 1.08, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake2'): [8, 1, 2.002/3600, 0.3, 1.65, 0, 1.35, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake3'): [8, 1, 2.228/3600, 0.24, 1.32, 0, 1.08, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake4'): [8, 1, 3.432/3600, 0.17, 0.93, 0, 0.77, 0],
        ('GCP', 'us-west1', 'n1-standard-8_v100_Fake5'): [8, 1, 4.290/3600, 0.14, 0.77, 0, 0.63, 0],
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
