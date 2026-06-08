import gurobipy as gp
import fl_data_model
from math import inf

B_round = 30000
T_round = 10000
server_msg_train = 0.537082008
server_msg_test = 0.537082008
client_msg_train = 0.537082008
client_msg_test = 0.000001808
alpha = 0.5
comm_baseline = 27.26

# Base data
# clients, baseline_exec, location = gp.multidict({
#     0: [233.01, 'us-central1'],
#     1: [233.01, 'us-central1']
# })
clients, baseline_exec, location = gp.multidict({
    0: [595.71, 'us-east-1'],
    1: [595.71, 'us-east-1']
})

print(f"2 clients")

for _ in range(1, 2):
    elapsed_time = fl_data_model.pre_process_model_vgg(clients=clients, location=location, baseline_exec=baseline_exec,
                                                       comm_baseline=comm_baseline, server_msg_train=server_msg_train,
                                                       server_msg_test=server_msg_test,
                                                       client_msg_train=client_msg_train,
                                                       client_msg_test=client_msg_test, T_round=T_round,
                                                       B_round=B_round, alpha=alpha)

    print(f"{elapsed_time},", end='')

print("\n----------------------------------------")

for client in range(2, 50):
    print(f"{client+1} clients")
    clients.append(client)
    # baseline_exec[client] = 233.01
    baseline_exec[client] = 595.71
    # location[client] = 'us-central1'
    location[client] = 'us-east-1'
    for _ in range(1, 2):

        elapsed_time = fl_data_model.pre_process_model_vgg(clients=clients, location=location,
                                                           baseline_exec=baseline_exec, comm_baseline=comm_baseline,
                                                           server_msg_train=server_msg_train,
                                                           server_msg_test=server_msg_test,
                                                           client_msg_train=client_msg_train,
                                                           client_msg_test=client_msg_test, T_round=T_round,
                                                           B_round=B_round, alpha=alpha)

        print(f"{elapsed_time},", end='')

    print("\n----------------------------------------")
