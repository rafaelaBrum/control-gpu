import gurobipy as gp
import fl_data_model


# Base data
clients, baseline_exec, location = gp.multidict({
    0: [608.40, 'us-east-1'],
    1: [608.40, 'us-east-1'],
    2: [608.40, 'us-east-1'],
    3: [233.01, 'us-central1']
})

B_round = 3000
T_round = 1000
server_msg_train = 0.537082008
server_msg_test = 0.537082008
client_msg_train = 0.537082008
client_msg_test = 0.000001808
alpha = 0.5
comm_baseline = 46.37

elapsed_time = fl_data_model.pre_process_model_vgg(clients=clients, location=location, baseline_exec=baseline_exec,
                                                   comm_baseline=comm_baseline, server_msg_train=server_msg_train,
                                                   server_msg_test=server_msg_test, client_msg_train=client_msg_train,
                                                   client_msg_test=client_msg_test, T_round=T_round, B_round=B_round,
                                                   alpha=alpha)

print(f"Gurobi elapsed time:{elapsed_time}")