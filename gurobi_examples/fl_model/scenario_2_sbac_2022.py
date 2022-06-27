import gurobipy as gp
import fl_data_model


# Base data
clients, baseline_exec, location = gp.multidict({
    0: [233.01, 'us-central1'],
    1: [233.01, 'us-central1'],
    2: [233.01, 'us-central1'],
    3: [233.01, 'us-central1']
    # ,
    # 4: [233.01, 'us-central1'],
    # 5: [233.01, 'us-central1'],
    # 6: [233.01, 'us-central1'],
    # 7: [233.01, 'us-central1'],
    # 8: [233.01, 'us-central1'],
    # 9: [233.01, 'us-central1'],
    # 10: [233.01, 'us-central1'],
    # 11: [233.01, 'us-central1'],
    # 12: [233.01, 'us-central1'],
    # 13: [233.01, 'us-central1'],
    # 14: [233.01, 'us-central1'],
    # 15: [233.01, 'us-central1'],
    # 16: [233.01, 'us-central1'],
    # 17: [233.01, 'us-central1'],
    # 18: [233.01, 'us-central1'],
    # 19: [233.01, 'us-central1'],
    # 20: [233.01, 'us-central1'],
    # 21: [233.01, 'us-central1'],
    # 22: [233.01, 'us-central1'],
    # 23: [233.01, 'us-central1'],
    # 24: [233.01, 'us-central1'],
    # 25: [233.01, 'us-central1'],
    # 26: [233.01, 'us-central1'],
    # 27: [233.01, 'us-central1'],
    # 28: [233.01, 'us-central1'],
    # 29: [233.01, 'us-central1'],
    # 30: [233.01, 'us-central1'],
    # 31: [233.01, 'us-central1'],
    # 32: [233.01, 'us-central1'],
    # 33: [233.01, 'us-central1'],
    # 34: [233.01, 'us-central1'],
    # 35: [233.01, 'us-central1'],
    # 36: [233.01, 'us-central1'],
    # 37: [233.01, 'us-central1'],
    # 38: [233.01, 'us-central1'],
    # 39: [233.01, 'us-central1'],
    # 40: [233.01, 'us-central1'],
    # 41: [233.01, 'us-central1'],
    # 42: [233.01, 'us-central1'],
    # 43: [233.01, 'us-central1'],
    # 44: [233.01, 'us-central1'],
    # 45: [233.01, 'us-central1'],
    # 46: [233.01, 'us-central1'],
    # 47: [233.01, 'us-central1'],
    # 48: [233.01, 'us-central1'],
    # 49: [233.01, 'us-central1']
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
