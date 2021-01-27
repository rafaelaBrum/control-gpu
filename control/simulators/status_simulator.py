from control.simulators.distributions import Poisson

from control.managers.virtual_machine import VirtualMachine
from control.managers.cloud_manager import CloudManager

import time
import logging

import threading


class RevocationSim:

    def __init__(self, termination_rate):
        self.rate = termination_rate

        logging.info("<TerminationSim>: Termination rate: {}".format(
            self.rate,
        ))

        self.running = True

        self.threads = []

    def __run(self, vm: VirtualMachine, model: Poisson):
        while not vm.ready:
            time.sleep(1)

        while self.running:

            time.sleep(1)

            # if global_state is running check if a termination occurs
            if vm.state == CloudManager.RUNNING and vm.ready:
                if model.event_happened():
                    vm.terminate(delete_volume=False)
                    return

    def stop_simulation(self):
        logging.info("Stopping simulation..")
        self.running = False

        for thread in self.threads:
            thread.join()

        logging.info("Simulation stopped")

    def register_vm(self, vm: VirtualMachine):
        # create a new Models
        model = Poisson(plambda=self.rate)

        # create a thread to simulate the states of the new VM type
        thread = threading.Thread(target=self.__run, args=[vm, model])
        thread.start()
        self.threads.append(thread)
