from control.domain.instance_type import InstanceType
# from control.domain.task import Task

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine

import logging


def test_on_demand_virtual_machine(disk_name, loader):
    instance = InstanceType(
        provider=CloudManager.GCLOUD,
        instance_type='n1-standard-8',
        image_id='disk-ubuntu-flower-client-test',
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand': 0.001,
                'preemptible': 0.000031},
        memory=30,
        vcpu=8,
        ebs_device_name='/dev/sdb',
        count_gpu=1,
        gpu='nvidia-tesla-t4'
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand',
        loader=loader,
        disk_name=disk_name
    )

    status = vm.deploy(type_task='server')

    print("IP of running instances:")
    print(vm.get_instances_ip())

    # print("On-demand price of instance:", vm.price)

    if status:
        vm.prepare_vm(type_task='server')

        input("Enter to continue with VM termination")

        status = vm.terminate()

        if status:
            logging.info("<VirtualMachine {}>: Terminated with Success".format(vm.instance_id, status))


# def main():
#     parser = argparse.ArgumentParser(description='Control GPU - v. 0.0.1')
#
#     parser.add_argument('--vm_number', required=True)
#     # parser.add_argument('--input_path', help="Path where there are all input files", type=str, default=None)
#     # parser.add_argument('--task_file', help="task file name", type=str, default=None)
#     # parser.add_argument('--env_file', help="env file name", type=str, default=None)
#     # parser.add_argument('--deadline_seconds', help="deadline (seconds)", type=int, default=None)
#     #
#     # parser.add_argument('--log_file', help="log file name", type=str, default=None)
#     #
#     # parser.add_argument('--revocation_rate',
#     #                     help="Revocation rate of the spot VMs [0.0 - 1.0] (simulation-only parameter)", type=float,
#     #                     default=None)
#
#     parser.add_argument('--command', default='control')
#
#     loader = Loader(args=parser.parse_args())
#
#     print("Testing on demand VM")
#     test_on_demand_virtual_machine(args.vm_number, loader)
#
#
# if __name__ == "__main__":
#     main()
#     # print("Testing spot VM")
#     # test_preemptible_virtual_machine()
#     print("Test completed")
