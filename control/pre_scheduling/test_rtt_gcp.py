from time import time, sleep

import argparse
import subprocess


def test_rtt(instance_name, zone, project_id):
    # self.client.load_system_host_keys()
    # print(f"gcloud compute ssh --zone {zone} --project {self.gcp_config.project} {instance_name} --command=\"{command}\"")
    try:
        print(f"gcloud compute ssh --zone {zone} --project {project_id} {instance_name} --command=\"ls\" --quiet")
        subprocess.run(f"gcloud compute ssh --zone {zone} --project {project_id} {instance_name} --command=\"ls\" --quiet", 
                       shell=True, check=True, capture_output=True)

    except Exception as e:

        print(e.stdout)
        print(e.stderr)
        print(e.returncode)
    for x in range(5):

        try:
            t1 = time()

            subprocess.run(f"gcloud compute ssh --zone {zone} --project {project_id} {instance_name} --command=\"ls\"", 
                           shell=True, check=True, capture_output=True)

            # time when connection is made
            t2 = time()

            return str(t2-t1)

        except Exception as e:

            print(e.stdout)
            print(e.stderr)
            print(e.returncode)

            sleep(10)

    return "-1"


def main():
    parser = argparse.ArgumentParser(description='Getting RTT of GCP instance')

    parser.add_argument('--instance_name', help="Instance name to connect", type=str, default=None, required=True)
    parser.add_argument('--zone', help="Zone in which the instance is located", type=str, default=None, required=True)
    parser.add_argument('--project_id', help="GCP project in which the instance is located", type=str, default=None, required=True)

    args = parser.parse_args()

    instance_name = args.instance_name
    zone = args.zone
    project_id = args.project_id
    rtt_time = test_rtt(instance_name, zone, project_id)
    print("RTT: ", rtt_time)


if __name__ == '__main__':
    main()