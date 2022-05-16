import flmodel
import gurobipy as gp
from math import inf

# Base data
clients = [0, 1, 2, 3]

B_round = 30
T_round = 900
server_msg_train = 1.07
server_msg_test = 0.54
client_msg_train = 1.07
client_msg_test = 0.54
alpha_1 = 0.5
alpha_2 = 0.5

providers, global_cpu_limits, global_gpu_limits, cost_transfer = gp.multidict({
    'AWS': [inf, inf, 0.090],
    'GCP': [4, 40, 0.120]
})

regions_prov, regional_cpu_limits, regional_gpu_limits = gp.multidict({
    ('AWS', 'us-east-1'): [40, inf],
    ('GCP', 'us-central1'): [40, 4]
})

vms_region_prov, cpu_vms, gpu_vms, cost_vms, time_aggreg = gp.multidict({
    ('AWS', 'us-east-1', 'g4dn.2xlarge'): [8, 1,  0.752/60, 1],
    ('AWS', 'us-east-1', 't2.xlarge'): [4, 0,  0.1856/60, 1],
    ('GCP', 'us-central1', 'n1-standard-8'): [8, 1,  0.730/60, 1],
    ('GCP', 'us-central1', 'e2-standard-4'): [4, 0,  0.134/60, 1]
})

client_prov_regions_vms, time_exec = gp.multidict({
    (0, 'AWS', 'us-east-1', 'g4dn.2xlarge'): 497,
    (1, 'AWS', 'us-east-1', 'g4dn.2xlarge'): 455,
    (2, 'AWS', 'us-east-1', 'g4dn.2xlarge'): 439,
    (3, 'AWS', 'us-east-1', 'g4dn.2xlarge'): 413,
    (0, 'GCP', 'us-central1', 'n1-standard-8'): 506,
    (1, 'GCP', 'us-central1', 'n1-standard-8'): 458,
    (2, 'GCP', 'us-central1', 'n1-standard-8'): 509,
    (3, 'GCP', 'us-central1', 'n1-standard-8'): 473,
})

pair_regions, time_comm = gp.multidict({
    ('AWS', 'us-east-1', 'AWS', 'us-east-1'): 4.485,
    ('AWS', 'us-east-1', 'GCP', 'us-central1'): 23.485,
    ('GCP', 'us-central1', 'GCP', 'us-central1'): 1.761,
    ('GCP', 'us-central1', 'AWS', 'us-east-1'): 23.485
})

flmodel.solve(client_prov_regions_vms=client_prov_regions_vms, cost_transfer=cost_transfer,
              vms_region_prov=vms_region_prov, cost_vms=cost_vms, server_msg_train=server_msg_train,
              server_msg_test=server_msg_test, client_msg_train=client_msg_train, client_msg_test=client_msg_test,
              T_round=T_round, clients=clients, B_round=B_round, alpha_1=alpha_1, alpha_2=alpha_2, providers=providers,
              gpu_vms=gpu_vms, global_gpu_limits=global_gpu_limits, cpu_vms=cpu_vms,
              global_cpu_limits=global_cpu_limits, regional_gpu_limits=regional_gpu_limits, regions_prov=regions_prov,
              regional_cpu_limits=regional_cpu_limits, time_exec=time_exec, time_comm=time_comm, time_aggreg=time_aggreg)