from datetime import datetime
from pathlib import Path

test_folder = 'Initial_mapping'
test_case = "4_clients"
current_exec = 8
n_rounds = 10
# -1 if there is no server ckpt
server_checkpoint = -1

n_clients = 4

first_framework_log_date_string = "2022-12-29 10:41:03,326"
last_framework_log_date_string = "2022-12-29 10:41:03,326"

first_server_message_date_string = "2023-04-26 12:39:44,285"
last_server_message_date_string = "2023-04-26 13:04:31,331"


def get_server_times_one_round(current_round, checkpoint):
    file_path = f'{Path.home()}/{test_folder}/' \
                f'{test_case}/Exec {current_exec}/server/screen_task_log_modified'

    initial_line = 0

    file = open(file_path, 'r')
    content = file.readlines()
    file = open(file_path, 'r')
    for pos, line in enumerate(file):
        if current_round > 0 and f"Round {current_round}\n" in line:
            initial_line = pos
        if current_round < 0 and f"Final" in line:
            initial_line = pos

    initial_line = initial_line + 2

    if checkpoint < 0:
        if current_round > 0:
            time1 = content[initial_line][13:36]
            gap = 4*n_clients+2+1
            time2 = content[initial_line + gap][13:35]
        else:
            time1 = content[initial_line][13:35]
            gap = 2*n_clients+1
            time2 = content[initial_line + gap][13:35]
    else:
        if current_round > 0:
            if current_round % checkpoint == 0:
                time1 = content[initial_line][13:36]
                gap = 4*n_clients+2+1+1
                time2 = content[initial_line + gap][13:35]
            else:
                time1 = content[initial_line][13:36]
                gap = 4*n_clients+2+1
                time2 = content[initial_line + gap][13:35]
        else:
            time1 = content[initial_line][13:35]
            gap = 2*n_clients+1
            time2 = content[initial_line + gap][13:35]

    # print("First time:", time1, "'")
    # print("Second time:", time2, "'")

    return time1, time2


def get_client_times_one_round(current_client, current_round):
    file_path = f'{Path.home()}/{test_folder}/' \
                f'{test_case}/Exec {current_exec}/client {current_client}/screen_task_log_modified'

    initial_line = 0

    file = open(file_path, 'r')
    content = file.readlines()
    file = open(file_path, 'r')
    for pos, line in enumerate(file):
        if current_round > 0 and f"Round {current_round}\n" in line:
            initial_line = pos
        if current_round < 0 and f"Final" in line:
            initial_line = pos

    if current_round > 0:
        time1 = content[initial_line + 2][12:35]
        time2 = content[initial_line + 13][12:35]
        time3 = content[initial_line + 14][12:35]
        time4 = content[initial_line + 17][12:35]
    else:
        time1 = content[initial_line + 2][12:35]
        time2 = time1
        time3 = time1
        time4 = content[initial_line + 5][12:35]

    # print("Client", current_client)
    # print("First time:", time1, "'")
    # print("Second time:", time2, "'")
    # print("Third time:", time3, "'")
    # print("Fourth time:", time4, "'")

    return time1, time2, time3, time4


if __name__ == "__main__":

    print(f'{Path.home()}/{test_folder}/{test_case}/Exec {current_exec}')

    for round_id in range(1, n_rounds+1):
        results = get_server_times_one_round(current_round=round_id, checkpoint=server_checkpoint)
        initial_sync_initial_date_string = results[0]
        test_sync_final_date_string = results[1]

        initial_sync_final_date_string = []
        aggregation_sync_initial_date_string = []
        aggregation_sync_final_date_string = []
        test_sync_initial_date_string = []

        for client in range(n_clients):
            results = get_client_times_one_round(client, round_id)
            initial_sync_final_date_string.append(results[0])
            aggregation_sync_initial_date_string.append(results[1])
            aggregation_sync_final_date_string.append(results[2])
            test_sync_initial_date_string.append(results[3])

        # Initial sync - Time between the server sending the first message and the last client receiving it
        initial_sync_final_date = []
        initial_sync_initial_date = datetime.strptime(initial_sync_initial_date_string, "%Y-%m-%d %H:%M:%S,%f")
        for date in initial_sync_final_date_string:
            initial_sync_final_date.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f"))

        initial_sync_duration = max(initial_sync_final_date) - initial_sync_initial_date

        # Aggregation sync - Time between the first client sending the updated weights and
        # the last client receiving the aggregated weights
        aggregation_sync_initial_date = []
        for date in aggregation_sync_initial_date_string:
            aggregation_sync_initial_date.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f"))

        aggregation_sync_final_date = []
        for date in aggregation_sync_final_date_string:
            aggregation_sync_final_date.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f"))

        aggregation_sync_duration = max(aggregation_sync_final_date) - min(aggregation_sync_initial_date)

        # Test sync - Time between the first client sending the test metrics and the server aggregates all
        test_sync_initial_date = []
        for date in test_sync_initial_date_string:
            test_sync_initial_date.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S,%f"))
        test_sync_final_date = datetime.strptime(test_sync_final_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_duration = test_sync_final_date - min(test_sync_initial_date)

        # Clients training and test times
        initial_training_time_date = []
        final_training_time_date = []
        initial_test_time_date = []
        final_test_time_date = []
        training_duration = []
        test_duration = []

        for client in range(n_clients):
            initial_training_time_date.append(initial_sync_final_date[client])
            final_training_time_date.append(aggregation_sync_initial_date[client])
            initial_test_time_date.append(aggregation_sync_final_date[client])
            final_test_time_date.append(test_sync_initial_date[client])
            training_duration.append(final_training_time_date[client] - initial_training_time_date[client])
            test_duration.append(final_test_time_date[client] - initial_test_time_date[client])

        # if round_id > 0:
        #     print(f'----------\n Round {round_id}\n----------')
        # else:
        #     print(f'----------\n Final\n----------')

        print_string = f"{initial_sync_duration}"
        for client in range(n_clients):
            print_string = print_string + f", {training_duration[client]}"
        print_string = print_string + f", {aggregation_sync_duration}"
        for client in range(n_clients):
            print_string = print_string + f", {test_duration[client]}"
        print_string = print_string + f", {test_sync_duration}"

        print(print_string)

    first_server_message_date = datetime.strptime(first_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
    last_server_message_date = datetime.strptime(last_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
    total_fl_execution_duration = last_server_message_date - first_server_message_date

    first_framework_log_date = datetime.strptime(first_framework_log_date_string, "%Y-%m-%d %H:%M:%S,%f")
    last_framework_log_date = datetime.strptime(last_framework_log_date_string, "%Y-%m-%d %H:%M:%S,%f")
    total_framework_execution_duration = last_framework_log_date - first_framework_log_date

    print(f'----------\n Total Exec Times\n----------')
    print(f"{total_fl_execution_duration}, {total_framework_execution_duration}")
