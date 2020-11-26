from control.domain.instance_type import InstanceType
from control.domain.task import Task

from control.managers.cloud_manager import CloudManager
from control.managers.virtual_machine import VirtualMachine


def test_virtual_machine():
    instance = InstanceType(
        provider=CloudManager.EC2,
        instance_type='t2.nano',
        image_id='ami-0d1a4eacad59b7a5b',
        memory=0.5,
        vcpu=1,
        restrictions={'on-demand': 1,
                      'preemptible': 1},
        prices={'on-demand':0.001,
                'preemptible':0.000031},
        gflops=0.0
    )

    task = Task(
        task_id=2,
        memory=0.2,
        command="ls",
        io=0,
        runtime={'t2.nano': 100}
    )

    vm = VirtualMachine(
        instance_type=instance,
        market='on-demand'
    )

    vm.deploy()

    vm.prepare_vm()

    vm.terminate()


if __name__ == "__main__":
    test_virtual_machine()
