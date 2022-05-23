import gurobipy as gp
import fl_data_model

B_round = 30000
T_round = 10000
server_msg_train = 0.537082008
server_msg_test = 0.537082008
client_msg_train = 0.537082008
client_msg_test = 0.000001808
alpha = 0.5
comm_baseline = 46.37

# Base data
clients, baseline_exec, location = gp.multidict({
    0: [608.40, 'us-east-1'],
    1: [608.40, 'us-east-1']
})

print(f"2 clients")

for _ in range(1, 11):
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
    baseline_exec[client] = 608.40
    location[client] = 'us-east-1'

    for _ in range(1, 11):

        elapsed_time = fl_data_model.pre_process_model_vgg(clients=clients, location=location,
                                                           baseline_exec=baseline_exec, comm_baseline=comm_baseline,
                                                           server_msg_train=server_msg_train,
                                                           server_msg_test=server_msg_test,
                                                           client_msg_train=client_msg_train,
                                                           client_msg_test=client_msg_test, T_round=T_round,
                                                           B_round=B_round, alpha=alpha)

        print(f"{elapsed_time},", end='')

    print("\n----------------------------------------")
