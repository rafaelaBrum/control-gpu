from datetime import datetime

first_date_string = "2023-02-23 16:31:30,543"
last_date_string = "2023-02-23 21:29:49,286"

price_c220 = 0.500928

price_c240 = 1.40784

price_r7525 = 3.347712

price_r320 = 0.184944

price = price_r320

if __name__ == "__main__":

    first_date = datetime.strptime(first_date_string, "%Y-%m-%d %H:%M:%S,%f")
    last_date = datetime.strptime(last_date_string, "%Y-%m-%d %H:%M:%S,%f")
    vm_execution = last_date - first_date

    print(f'VM Exec Time: {vm_execution} - Price: {price*(vm_execution.total_seconds()/3600)}')
