import paramiko
import socket

from time import time, sleep

import argparse


def test_rtt(ip_address, key_path, key_file, user):
    client = paramiko.SSHClient()
    # self.client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    for x in range(5):

        try:
            t1 = time()

            client.connect(
                hostname=ip_address,
                port=22,
                username=user,
                pkey=paramiko.RSAKey.from_private_key_file(key_path + key_file),
                timeout=30
            )

            # time when connection is made
            t2 = time()

            client.close()

            return str(t2-t1)

        except (paramiko.BadHostKeyException, paramiko.AuthenticationException,
                paramiko.SSHException, socket.error) as e:

            print(e)

            sleep(10)

    return "-1"


def main():
    parser = argparse.ArgumentParser(description='Getting RTT of instance')

    parser.add_argument('--ip', help="IP to testing instance", type=str, default=None, required=True)
    parser.add_argument('--path', help="Path inside VM where the key is located", type=str, default=None, required=True)
    parser.add_argument('--file', help="Key file name", type=str, default=None, required=True)
    parser.add_argument('--user', help="VM user to connect to the instance", type=str, default=None, required=True)

    args = parser.parse_args()

    ip = args.ip
    path = args.path
    file = args.file
    user_cloud = args.user
    rtt_time = test_rtt(ip, path, file, user_cloud)
    print("RTT: ", rtt_time)


if __name__ == '__main__':
    main()

# >>> test = "RTT: -1"
# >>> test.split(" ")
# ['RTT:', '-1']
# >>> test_div = test.split(" ")
# >>> test_div[-1]
# '-1'
# >>> int(test_div[-1])
# -1
# >>> float(test_div[-1])
# -1.0
# >>> test = "RTT: 0.23256845645645"
# >>> test_div = test.split(" ")
# >>> float(test_div[-1])
# 0.23256845645645
