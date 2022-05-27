from datetime import datetime
from pathlib import Path

test_folder = 'Testes_SBAC_2022/Sc1'
test_case = "Optimal"
current_exec = 1

first_server_message_date_string = "2022-05-24 13:53:42,846"
last_server_message_date_string = "2022-05-24 15:24:52,818"


def get_server_times_one_round(n_clients, current_round):
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

    if current_round > 0:
        time1 = content[initial_line][13:36]
        gap = 4*n_clients+2+1
        time2 = content[initial_line + gap][13:36]
    else:
        time1 = content[initial_line][13:36]
        gap = 2*n_clients+1
        time2 = content[initial_line + gap][13:36]

    # print("First time:", time1)
    # print("Second time:", time2)
    # print("Third time:", time3)
    # print("Fourth time:", time4)

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

    # print("First time:", time1)
    # print("Second time:", time2)
    # print("Third time:", time3)
    # print("Fourth time:", time4)

    return time1, time2, time3, time4


def put_times_by_hand():
    # Initial sync - Time between the server sending the first message and the last client receiving it
    initial_sync_initial_date_string = "2022-04-11 23:04:17,042"
    initial_sync_final_client_0_date_string = "2022-04-11 23:04:30,018"
    initial_sync_final_client_1_date_string = "2022-04-11 23:04:31,359"
    initial_sync_final_client_2_date_string = "2022-04-11 23:04:21,606"
    initial_sync_final_client_3_date_string = "2022-04-11 23:04:21,935"

    # Aggregation sync - Time between the first client sending the updated weights and
    # the last client receiving the aggregated weights
    aggregation_sync_initial_client_0_date_string = "2022-04-11 23:09:28,038"
    aggregation_sync_initial_client_1_date_string = "2022-04-11 23:10:20,881"
    aggregation_sync_initial_client_2_date_string = "2022-04-11 23:06:39,736"
    aggregation_sync_initial_client_3_date_string = "2022-04-11 23:06:42,332"
    aggregation_sync_final_client_0_date_string = "2022-04-11 23:10:48,115"
    aggregation_sync_final_client_1_date_string = "2022-04-11 23:10:43,396"
    aggregation_sync_final_client_2_date_string = "2022-04-11 23:10:33,440"
    aggregation_sync_final_client_3_date_string = "2022-04-11 23:10:31,984"

    # Test sync - Time between the first client sending the test metrics and the server aggregates all
    test_sync_initial_client_0_date_string = "2022-04-11 23:12:36,486"
    test_sync_initial_client_1_date_string = "2022-04-11 23:12:41,474"
    test_sync_initial_client_2_date_string = "2022-04-11 23:11:05,282"
    test_sync_initial_client_3_date_string = "2022-04-11 23:11:10,652"
    test_sync_final_date_string = "2022-04-11 23:12:41,558"


if __name__ == "__main__":

    for round in range(1, 11):
        results = get_server_times_one_round(n_clients=4, current_round=round)
        initial_sync_initial_date_string = results[0]
        test_sync_final_date_string = results[1]

        client = 0
        results = get_client_times_one_round(client, round)
        initial_sync_final_client_0_date_string = results[0]
        aggregation_sync_initial_client_0_date_string = results[1]
        aggregation_sync_final_client_0_date_string = results[2]
        test_sync_initial_client_0_date_string = results[3]
        client = 1
        results = get_client_times_one_round(client, round)
        initial_sync_final_client_1_date_string = results[0]
        aggregation_sync_initial_client_1_date_string = results[1]
        aggregation_sync_final_client_1_date_string = results[2]
        test_sync_initial_client_1_date_string = results[3]
        client = 2
        results = get_client_times_one_round(client, round)
        initial_sync_final_client_2_date_string = results[0]
        aggregation_sync_initial_client_2_date_string = results[1]
        aggregation_sync_final_client_2_date_string = results[2]
        test_sync_initial_client_2_date_string = results[3]
        client = 3
        results = get_client_times_one_round(client, round)
        initial_sync_final_client_3_date_string = results[0]
        aggregation_sync_initial_client_3_date_string = results[1]
        aggregation_sync_final_client_3_date_string = results[2]
        test_sync_initial_client_3_date_string = results[3]

        first_server_message_date = datetime.strptime(first_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
        last_server_message_date = datetime.strptime(last_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
        total_fl_execution_duration = last_server_message_date - first_server_message_date

        # Initial sync - Time between the server sending the first message and the last client receiving it
        initial_sync_initial_date = datetime.strptime(initial_sync_initial_date_string, "%Y-%m-%d %H:%M:%S,%f")
        initial_sync_final_client_0_date = datetime.strptime(initial_sync_final_client_0_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
        initial_sync_final_client_1_date = datetime.strptime(initial_sync_final_client_1_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
        initial_sync_final_client_2_date = datetime.strptime(initial_sync_final_client_2_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
        initial_sync_final_client_3_date = datetime.strptime(initial_sync_final_client_3_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
        initial_sync_duration = max(initial_sync_final_client_0_date,
                                    initial_sync_final_client_1_date,
                                    initial_sync_final_client_2_date,
                                    initial_sync_final_client_3_date) - initial_sync_initial_date

        # Aggregation sync - Time between the first client sending the updated weights and
        # the last client receiving the aggregated weights
        aggregation_sync_initial_client_0_date = datetime.strptime(aggregation_sync_initial_client_0_date_string,
                                                                   "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_initial_client_1_date = datetime.strptime(aggregation_sync_initial_client_1_date_string,
                                                                   "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_initial_client_2_date = datetime.strptime(aggregation_sync_initial_client_2_date_string,
                                                                   "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_initial_client_3_date = datetime.strptime(aggregation_sync_initial_client_3_date_string,
                                                                   "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_final_client_0_date = datetime.strptime(aggregation_sync_final_client_0_date_string,
                                                                 "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_final_client_1_date = datetime.strptime(aggregation_sync_final_client_1_date_string,
                                                                 "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_final_client_2_date = datetime.strptime(aggregation_sync_final_client_2_date_string,
                                                                 "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_final_client_3_date = datetime.strptime(aggregation_sync_final_client_3_date_string,
                                                                 "%Y-%m-%d %H:%M:%S,%f")
        aggregation_sync_duration = max(aggregation_sync_final_client_0_date,
                                        aggregation_sync_final_client_1_date,
                                        aggregation_sync_final_client_2_date,
                                        aggregation_sync_final_client_3_date) - min(aggregation_sync_initial_client_0_date,
                                                                                    aggregation_sync_initial_client_1_date,
                                                                                    aggregation_sync_initial_client_2_date,
                                                                                    aggregation_sync_initial_client_3_date)

        # Test sync - Time between the first client sending the test metrics and the server aggregates all
        test_sync_initial_client_0_date = datetime.strptime(test_sync_initial_client_0_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_initial_client_1_date = datetime.strptime(test_sync_initial_client_1_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_initial_client_2_date = datetime.strptime(test_sync_initial_client_2_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_initial_client_3_date = datetime.strptime(test_sync_initial_client_3_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_final_date = datetime.strptime(test_sync_final_date_string, "%Y-%m-%d %H:%M:%S,%f")
        test_sync_duration = test_sync_final_date - min(test_sync_initial_client_0_date,
                                                        test_sync_initial_client_1_date,
                                                        test_sync_initial_client_2_date,
                                                        test_sync_initial_client_3_date)

        # Clients training and test times
        initial_training_time_client_0_date = initial_sync_final_client_0_date
        initial_training_time_client_1_date = initial_sync_final_client_1_date
        initial_training_time_client_2_date = initial_sync_final_client_2_date
        initial_training_time_client_3_date = initial_sync_final_client_3_date
        final_training_time_client_0_date = aggregation_sync_initial_client_0_date
        final_training_time_client_1_date = aggregation_sync_initial_client_1_date
        final_training_time_client_2_date = aggregation_sync_initial_client_2_date
        final_training_time_client_3_date = aggregation_sync_initial_client_3_date

        initial_test_time_client_0_date = aggregation_sync_final_client_0_date
        initial_test_time_client_1_date = aggregation_sync_final_client_1_date
        initial_test_time_client_2_date = aggregation_sync_final_client_2_date
        initial_test_time_client_3_date = aggregation_sync_final_client_3_date
        final_test_time_client_0_date = test_sync_initial_client_0_date
        final_test_time_client_1_date = test_sync_initial_client_1_date
        final_test_time_client_2_date = test_sync_initial_client_2_date
        final_test_time_client_3_date = test_sync_initial_client_3_date

        training_client_0_duration = final_training_time_client_0_date - initial_training_time_client_0_date
        training_client_1_duration = final_training_time_client_1_date - initial_training_time_client_1_date
        training_client_2_duration = final_training_time_client_2_date - initial_training_time_client_2_date
        training_client_3_duration = final_training_time_client_3_date - initial_training_time_client_3_date

        test_client_0_duration = final_test_time_client_0_date - initial_test_time_client_0_date
        test_client_1_duration = final_test_time_client_1_date - initial_test_time_client_1_date
        test_client_2_duration = final_test_time_client_2_date - initial_test_time_client_2_date
        test_client_3_duration = final_test_time_client_3_date - initial_test_time_client_3_date

        # print("Initial sync duration: ", initial_sync_duration)
        #
        # print("Client 0 training duration:", training_client_0_duration)
        #
        # print("Client 1 training duration:", training_client_1_duration)
        #
        # print("Client 2 training duration:", training_client_2_duration)
        #
        # print("Client 3 training duration:", training_client_3_duration)
        #
        # print("Aggregation sync duration: ", aggregation_sync_duration)
        #
        # print("Client 0 test duration:", test_client_0_duration)
        #
        # print("Client 1 test duration:", test_client_1_duration)
        #
        # print("Client 2 test duration:", test_client_2_duration)
        #
        # print("Client 3 test duration:", test_client_3_duration)
        #
        # print("Test sync duration: ", test_sync_duration)
        #
        # print("FL total execution: ", total_fl_execution_duration)

        if round > 0:
            print(f'----------\n Round {round}\n----------')
        else:
            print(f'----------\n Final\n----------')

        print(f"{initial_sync_duration}, {training_client_0_duration}, {training_client_1_duration}, "
              f"{training_client_2_duration}, {training_client_3_duration}, {aggregation_sync_duration}, "
              f"{test_client_0_duration}, {test_client_1_duration}, {test_client_2_duration}, "
              f"{test_client_3_duration}, {test_sync_duration}, {total_fl_execution_duration}")

    round = -1

    results = get_server_times_one_round(n_clients=4, current_round=round)
    initial_sync_initial_date_string = results[0]
    test_sync_final_date_string = results[1]

    client = 0
    results = get_client_times_one_round(client, round)
    initial_sync_final_client_0_date_string = results[0]
    aggregation_sync_initial_client_0_date_string = results[1]
    aggregation_sync_final_client_0_date_string = results[2]
    test_sync_initial_client_0_date_string = results[3]
    client = 1
    results = get_client_times_one_round(client, round)
    initial_sync_final_client_1_date_string = results[0]
    aggregation_sync_initial_client_1_date_string = results[1]
    aggregation_sync_final_client_1_date_string = results[2]
    test_sync_initial_client_1_date_string = results[3]
    client = 2
    results = get_client_times_one_round(client, round)
    initial_sync_final_client_2_date_string = results[0]
    aggregation_sync_initial_client_2_date_string = results[1]
    aggregation_sync_final_client_2_date_string = results[2]
    test_sync_initial_client_2_date_string = results[3]
    client = 3
    results = get_client_times_one_round(client, round)
    initial_sync_final_client_3_date_string = results[0]
    aggregation_sync_initial_client_3_date_string = results[1]
    aggregation_sync_final_client_3_date_string = results[2]
    test_sync_initial_client_3_date_string = results[3]

    first_server_message_date = datetime.strptime(first_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
    last_server_message_date = datetime.strptime(last_server_message_date_string, "%Y-%m-%d %H:%M:%S,%f")
    total_fl_execution_duration = last_server_message_date - first_server_message_date

    # Initial sync - Time between the server sending the first message and the last client receiving it
    initial_sync_initial_date = datetime.strptime(initial_sync_initial_date_string, "%Y-%m-%d %H:%M:%S,%f")
    initial_sync_final_client_0_date = datetime.strptime(initial_sync_final_client_0_date_string,
                                                         "%Y-%m-%d %H:%M:%S,%f")
    initial_sync_final_client_1_date = datetime.strptime(initial_sync_final_client_1_date_string,
                                                         "%Y-%m-%d %H:%M:%S,%f")
    initial_sync_final_client_2_date = datetime.strptime(initial_sync_final_client_2_date_string,
                                                         "%Y-%m-%d %H:%M:%S,%f")
    initial_sync_final_client_3_date = datetime.strptime(initial_sync_final_client_3_date_string,
                                                         "%Y-%m-%d %H:%M:%S,%f")
    initial_sync_duration = max(initial_sync_final_client_0_date,
                                initial_sync_final_client_1_date,
                                initial_sync_final_client_2_date,
                                initial_sync_final_client_3_date) - initial_sync_initial_date

    # Aggregation sync - Time between the first client sending the updated weights and
    # the last client receiving the aggregated weights
    aggregation_sync_initial_client_0_date = datetime.strptime(aggregation_sync_initial_client_0_date_string,
                                                               "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_initial_client_1_date = datetime.strptime(aggregation_sync_initial_client_1_date_string,
                                                               "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_initial_client_2_date = datetime.strptime(aggregation_sync_initial_client_2_date_string,
                                                               "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_initial_client_3_date = datetime.strptime(aggregation_sync_initial_client_3_date_string,
                                                               "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_final_client_0_date = datetime.strptime(aggregation_sync_final_client_0_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_final_client_1_date = datetime.strptime(aggregation_sync_final_client_1_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_final_client_2_date = datetime.strptime(aggregation_sync_final_client_2_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_final_client_3_date = datetime.strptime(aggregation_sync_final_client_3_date_string,
                                                             "%Y-%m-%d %H:%M:%S,%f")
    aggregation_sync_duration = max(aggregation_sync_final_client_0_date,
                                    aggregation_sync_final_client_1_date,
                                    aggregation_sync_final_client_2_date,
                                    aggregation_sync_final_client_3_date) - min(aggregation_sync_initial_client_0_date,
                                                                                aggregation_sync_initial_client_1_date,
                                                                                aggregation_sync_initial_client_2_date,
                                                                                aggregation_sync_initial_client_3_date)

    # Test sync - Time between the first client sending the test metrics and the server aggregates all
    test_sync_initial_client_0_date = datetime.strptime(test_sync_initial_client_0_date_string, "%Y-%m-%d %H:%M:%S,%f")
    test_sync_initial_client_1_date = datetime.strptime(test_sync_initial_client_1_date_string, "%Y-%m-%d %H:%M:%S,%f")
    test_sync_initial_client_2_date = datetime.strptime(test_sync_initial_client_2_date_string, "%Y-%m-%d %H:%M:%S,%f")
    test_sync_initial_client_3_date = datetime.strptime(test_sync_initial_client_3_date_string, "%Y-%m-%d %H:%M:%S,%f")
    test_sync_final_date = datetime.strptime(test_sync_final_date_string, "%Y-%m-%d %H:%M:%S,%f")
    test_sync_duration = test_sync_final_date - min(test_sync_initial_client_0_date,
                                                    test_sync_initial_client_1_date,
                                                    test_sync_initial_client_2_date,
                                                    test_sync_initial_client_3_date)

    # Clients training and test times
    initial_training_time_client_0_date = initial_sync_final_client_0_date
    initial_training_time_client_1_date = initial_sync_final_client_1_date
    initial_training_time_client_2_date = initial_sync_final_client_2_date
    initial_training_time_client_3_date = initial_sync_final_client_3_date
    final_training_time_client_0_date = aggregation_sync_initial_client_0_date
    final_training_time_client_1_date = aggregation_sync_initial_client_1_date
    final_training_time_client_2_date = aggregation_sync_initial_client_2_date
    final_training_time_client_3_date = aggregation_sync_initial_client_3_date

    initial_test_time_client_0_date = aggregation_sync_final_client_0_date
    initial_test_time_client_1_date = aggregation_sync_final_client_1_date
    initial_test_time_client_2_date = aggregation_sync_final_client_2_date
    initial_test_time_client_3_date = aggregation_sync_final_client_3_date
    final_test_time_client_0_date = test_sync_initial_client_0_date
    final_test_time_client_1_date = test_sync_initial_client_1_date
    final_test_time_client_2_date = test_sync_initial_client_2_date
    final_test_time_client_3_date = test_sync_initial_client_3_date

    training_client_0_duration = final_training_time_client_0_date - initial_training_time_client_0_date
    training_client_1_duration = final_training_time_client_1_date - initial_training_time_client_1_date
    training_client_2_duration = final_training_time_client_2_date - initial_training_time_client_2_date
    training_client_3_duration = final_training_time_client_3_date - initial_training_time_client_3_date

    test_client_0_duration = final_test_time_client_0_date - initial_test_time_client_0_date
    test_client_1_duration = final_test_time_client_1_date - initial_test_time_client_1_date
    test_client_2_duration = final_test_time_client_2_date - initial_test_time_client_2_date
    test_client_3_duration = final_test_time_client_3_date - initial_test_time_client_3_date

    # print("Initial sync duration: ", initial_sync_duration)
    #
    # print("Client 0 training duration:", training_client_0_duration)
    #
    # print("Client 1 training duration:", training_client_1_duration)
    #
    # print("Client 2 training duration:", training_client_2_duration)
    #
    # print("Client 3 training duration:", training_client_3_duration)
    #
    # print("Aggregation sync duration: ", aggregation_sync_duration)
    #
    # print("Client 0 test duration:", test_client_0_duration)
    #
    # print("Client 1 test duration:", test_client_1_duration)
    #
    # print("Client 2 test duration:", test_client_2_duration)
    #
    # print("Client 3 test duration:", test_client_3_duration)
    #
    # print("Test sync duration: ", test_sync_duration)
    #
    # print("FL total execution: ", total_fl_execution_duration)

    if round > 0:
        print(f'----------\n Round {round}\n----------')
    else:
        print(f'----------\n Final\n----------')

    print(f"{initial_sync_duration}, {training_client_0_duration}, {training_client_1_duration}, "
          f"{training_client_2_duration}, {training_client_3_duration}, {aggregation_sync_duration}, "
          f"{test_client_0_duration}, {test_client_1_duration}, {test_client_2_duration}, "
          f"{test_client_3_duration}, {test_sync_duration}, {total_fl_execution_duration}")
