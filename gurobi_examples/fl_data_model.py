import flmodel
import gurobipy as gp
from math import inf

# Base data
clients, baseline_exec, location = gp.multidict({
    0: [1050.30, 'us-east-1'],
    1: [913.23, 'us-east-1'],
    2: [890.43, 'us-east-1'],
    3: [814.33, 'us-east-1']
})

B_round = 3000
T_round = 9000
server_msg_train = 1.07
server_msg_test = 0.54
client_msg_train = 1.07
client_msg_test = 0.54
alpha_1 = 0.5
alpha_2 = 0.5
comm_baseline = 46.37

providers, global_cpu_limits, global_gpu_limits, cost_transfer = gp.multidict({
    'AWS': [inf, inf, 0.090],
    'GCP': [4, 40, 0.120]
})

prov_regions, regional_cpu_limits, regional_gpu_limits = gp.multidict({
    ('AWS', 'us-east-1'): [48, inf],
    ('AWS', 'us-west-2'): [32, inf],
    ('GCP', 'us-central1'): [24, 4],
    ('GCP', 'us-west1'): [24, 4]
})

prov_regions_vms, cpu_vms, gpu_vms, cost_vms, \
time_aggreg, slowdown_us_east_1, slowdown_us_west_2, slowdown_us_central1, slowdown_us_west1 = gp.multidict({
    ('AWS', 'us-east-1', 'g4dn.2xlarge'): [8, 1,  0.752/60, 0.3, 1.0, 0, 0, 0],
    ('AWS', 'us-east-1', 'g3.4xlarge'): [16, 1,  1.14/60, 0.3, 4.98, 0, 0, 0],
    # ('AWS', 'us-east-1', 'p3.2xlarge'): [16, 1,  3.06/60, 0.3, 4.82, 0, 0, 0],
    ('AWS', 'us-east-1', 't2.xlarge'): [4, 0,  0.1856/60, 0.3, 10, 10, 10, 10],
    ('AWS', 'us-west-2', 'g4dn.2xlarge'): [8, 1,  0.752/60, 0.3, 0.85, 0, 0, 0],
    ('AWS', 'us-west-2', 'g3.4xlarge'): [16, 1,  1.14/60, 0.3, 4.35, 0, 0, 0],
    ('AWS', 'us-west-2', 't2.xlarge'): [4, 0,  0.1856/60, 0.3, 10, 10, 10, 10],
    ('GCP', 'us-central1', 'n1-standard-8_t4'): [8, 1,  0.730/60, 0.3, 1.01, 0, 0, 0],
    ('GCP', 'us-central1', 'n1-standard-8_p4'): [8, 1,  1.360/60, 0.3, 1.25, 0, 0, 0],
    ('GCP', 'us-central1', 'n1-standard-8_v100'): [8, 1,  2.860/60, 0.3, 1.01, 0, 0, 0],
    ('GCP', 'us-central1', 'e2-standard-4'): [4, 0,  0.134/60, 0.3, 10, 10, 10, 10],
    ('GCP', 'us-west1', 'n1-standard-8_t4'): [8, 1,  0.730/60, 0.3, 1.04, 0, 0, 0],
    ('GCP', 'us-west1', 'n1-standard-8_v100'): [8, 1,  2.860/60, 0.3, 1.08, 0, 0, 0],
    ('GCP', 'us-west1', 'e2-standard-4'): [4, 0,  0.134/60, 0.3, 10, 10, 10, 10]
})

client_prov_regions_vms = []
time_exec = {}
for client in clients:
    for prov, region, vm in prov_regions_vms:
        aux = (client, prov, region, vm)
        # print(aux)
        client_prov_regions_vms.append(aux)
        if location[client] == 'us-east-1':
            time_exec[aux] = baseline_exec[client]*slowdown_us_east_1[prov, region, vm]
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

flmodel.solve(client_prov_regions_vms=client_prov_regions_vms, cost_transfer=cost_transfer,
              prov_regions_vms=prov_regions_vms, cost_vms=cost_vms, server_msg_train=server_msg_train,
              server_msg_test=server_msg_test, client_msg_train=client_msg_train, client_msg_test=client_msg_test,
              T_round=T_round, clients=clients, B_round=B_round, alpha_1=alpha_1, alpha_2=alpha_2, providers=providers,
              gpu_vms=gpu_vms, global_gpu_limits=global_gpu_limits, cpu_vms=cpu_vms,
              global_cpu_limits=global_cpu_limits, regional_gpu_limits=regional_gpu_limits,
              prov_regions=prov_regions, regional_cpu_limits=regional_cpu_limits, time_exec=time_exec,
              time_comm=time_comm, time_aggreg=time_aggreg)