#!/usr/bin/env python

from control.util.loader import Loader

from control.tests.test_virtual_machine_on_gcp import test_on_demand_virtual_machine as test_gcp_on_demand
from control.tests.test_virtual_machine import test_on_demand_virtual_machine as test_aws_on_demand

import argparse
# import datetime


def main():
    parser = argparse.ArgumentParser(description='Control GPU - v. 0.0.1')

    parser.add_argument('--vm_number', required=True)
    parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
    parser.add_argument('--task_file', help="task file name", type=str, default=None)
    parser.add_argument('--env_file', help="env file name", type=str, default=None)
    parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)

    parser.add_argument('--log_file', help="log file name", type=str, default=None)

    parser.add_argument('--revocation_rate',
                        help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
                        default=None)

    parser.add_argument('--command', default='control')

    args = parser.parse_args()

    loader = Loader(args=args)

    print("Testing on-demand VM on AWS")
    test_aws_on_demand(loader)

    print("Testing on-demand VM on GCP")
    test_gcp_on_demand(args.vm_number, loader)


if __name__ == "__main__":
    main()
