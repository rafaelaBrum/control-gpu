import gurobipy as gp
import fl_data_model


# Base data

clients, baseline_exec, location = gp.multidict({
    0: [595.71, 'us-east-1'],
    1: [595.71, 'us-east-1'],
    2: [595.71, 'us-east-1'],
    3: [595.71, 'us-east-1']
    # ,
    # 4: [595.71, 'us-east-1'],
    # 5: [595.71, 'us-east-1'],
    # 6: [595.71, 'us-east-1'],
    # 7: [595.71, 'us-east-1'],
    # 8: [595.71, 'us-east-1'],
    # 9: [595.71, 'us-east-1'],
    # 10: [595.71, 'us-east-1'],
    # 11: [595.71, 'us-east-1'],
    # 12: [595.71, 'us-east-1'],
    # 13: [595.71, 'us-east-1'],
    # 14: [595.71, 'us-east-1'],
    # 15: [595.71, 'us-east-1'],
    # 16: [595.71, 'us-east-1'],
    # 17: [595.71, 'us-east-1'],
    # 18: [595.71, 'us-east-1'],
    # 19: [595.71, 'us-east-1']
})

B_round = 3000
T_round = 1000
server_msg_train = 0.537082008
server_msg_test = 0.537082008
client_msg_train = 0.537082008
client_msg_test = 0.000001808
alpha = 0.5
comm_baseline = 27.26

print(f"alpha = {alpha}")

elapsed_time = fl_data_model.pre_process_model_vgg(clients=clients, location=location, baseline_exec=baseline_exec,
                                                   comm_baseline=comm_baseline, server_msg_train=server_msg_train,
                                                   server_msg_test=server_msg_test, client_msg_train=client_msg_train,
                                                   client_msg_test=client_msg_test, T_round=T_round, B_round=B_round,
                                                   alpha=alpha)

print(f"Gurobi elapsed time:{elapsed_time}")
