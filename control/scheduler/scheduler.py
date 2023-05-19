from control.domain.instance_type import InstanceType
from control.domain.cloud_region import CloudRegion

from control.managers.cloud_manager import CloudManager
from control.managers.experiment_cloudlab import Experiment

from typing import Dict

import logging


class Scheduler:
    FL_SIMPLE = "SIMPLE"
    MAT_FORM = "FORMULATION"
    DYNAMIC_GREEDY = "GREEDY"

    def __init__(self, instance_types: Dict[str, InstanceType], locations: Dict[str, CloudRegion]):
        self.instances_server_aws: Dict[str, InstanceType] = {}
        self.instances_server_gcp: Dict[str, InstanceType] = {}
        self.instances_server_cloudlab: Dict[str, InstanceType] = {}
        self.instances_client_gcp: Dict[str, InstanceType] = {}
        self.instances_client_aws: Dict[str, InstanceType] = {}
        self.instances_client_cloudlab: Dict[str, InstanceType] = {}
        self.loc_aws: Dict[str, CloudRegion] = {}
        self.loc_gcp: Dict[str, CloudRegion] = {}
        self.loc_cloudlab: Dict[str, CloudRegion] = {}
        self.__divide_instances_for_server_and_for_client_by_cloud(instance_types)
        self.__separate_location_by_cloud(locations)
        self.index_extra_vm = []
        self.current_vms: Dict[str, InstanceType] = {'server': None}
        self.current_locations: Dict[str, CloudRegion] = {'server': None}
        self.qtde_gpus_spot_gcloud = 0
        self.client_id_spot_gpu = -1

    def __divide_instances_for_server_and_for_client_by_cloud(self, instance_types):
        # logging.info("<Scheduler>: Dividing instances types for server and client")

        for name, instance in instance_types.items():
            # logging.info("<Scheduler>: Instance type {} has GPU? {}".format(name, instance.have_gpu))
            if instance.provider in CloudManager.CLOUDLAB:
                self.instances_server_cloudlab[name] = instance
                self.instances_client_cloudlab[name] = instance
            elif instance.have_gpu:
                if instance.provider in (CloudManager.EC2, CloudManager.AWS):
                    self.instances_client_aws[name] = instance
                    # logging.info("<Scheduler>: Instance type {} added to instances_client_aws".format(name))
                elif instance.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                    self.instances_client_gcp[name] = instance
                    # logging.info("<Scheduler>: Instance type {} added to instances_client_gcp".format(name))
                else:
                    logging.error(f"<Scheduler>: {instance.provider} does not have support ({name})")
            else:
                if instance.provider in (CloudManager.EC2, CloudManager.AWS):
                    self.instances_server_aws[name] = instance
                    # logging.info("<Scheduler>: Instance type {} added to instances_server_aws".format(name))
                elif instance.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                    self.instances_server_gcp[name] = instance
                    # logging.info("<Scheduler>: Instance type {} added to instances_server_gcp".format(name))
                else:
                    logging.error(f"<Scheduler>: {instance.provider} does not have support ({name})")

    def get_server_instance(self, provider, region, vm_name):
        logging.info("<Scheduler>: Choosing instance for server task from provider {} in region {} "
                     "with name {}".format(provider, region, vm_name))
        if provider.lower() in (CloudManager.EC2, CloudManager.AWS):
            for name, instance in self.instances_server_aws.items():
                if name == vm_name:
                    for loc in self.loc_aws.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                self.current_vms['server'] = instance
                                self.current_locations['server'] = loc
                                return instance, CloudManager.PREEMPTIBLE, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))
        elif provider.lower() in (CloudManager.GCLOUD, CloudManager.GCP):
            for name, instance in self.instances_server_gcp.items():
                if name == vm_name:
                    for loc in self.loc_gcp.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                self.current_vms['server'] = instance
                                self.current_locations['server'] = loc
                                return instance, CloudManager.ON_DEMAND, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))
        elif provider.lower() in (CloudManager.CLOUDLAB.lower()):
            # print("instances", self.instances_server_cloudlab)
            for name, instance in self.instances_server_cloudlab.items():
                # print("name", name)
                if name == vm_name:
                    for loc in self.loc_cloudlab.values():
                        # print("loc", loc)
                        if loc.region == region:
                            logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                            self.current_vms['server'] = instance
                            self.current_locations['server'] = loc
                            return instance, Experiment.MARKET, loc, ""
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))

    def get_client_instance(self, provider, region, vm_name, client_id):
        logging.info("<Scheduler>: Choosing instance for client task from provider {} in region {} "
                     "with name {}".format(provider, region, vm_name))
        if provider.lower() in (CloudManager.EC2, CloudManager.AWS):
            for name, instance in self.instances_client_aws.items():
                if name == vm_name:
                    for loc in self.loc_aws.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                self.current_vms[str(client_id)] = instance
                                self.current_locations[str(client_id)] = loc
                                return instance, CloudManager.PREEMPTIBLE, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))
        elif provider.lower() in (CloudManager.GCLOUD, CloudManager.GCP):
            for name, instance in self.instances_client_gcp.items():
                if name == vm_name:
                    for loc in self.loc_gcp.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                if self.qtde_gpus_spot_gcloud >= 1 and self.client_id_spot_gpu == client_id:
                                    self.qtde_gpus_spot_gcloud -= 1
                                    self.client_id_spot_gpu = -1
                                self.current_vms[str(client_id)] = instance
                                self.current_locations[str(client_id)] = loc
                                if self.qtde_gpus_spot_gcloud < 1:
                                    self.qtde_gpus_spot_gcloud += 1
                                    self.client_id_spot_gpu = client_id
                                    return instance, CloudManager.ON_DEMAND, loc, zone
                                else:
                                    return instance, CloudManager.ON_DEMAND, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
        elif provider.lower() in (CloudManager.CLOUDLAB.lower()):
            for name, instance in self.instances_client_cloudlab.items():
                if name == vm_name:
                    for loc in self.loc_cloudlab.values():
                        if loc.region == region:
                            logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                            self.current_vms[str(client_id)] = instance
                            self.current_locations[str(client_id)] = loc
                            return instance, Experiment.MARKET, loc, ""
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))

    def get_extra_vm_instance(self, provider, region):
        logging.info("<Scheduler>: Choosing initial instance for extra VM from provider {}".format(provider))
        if provider.lower() in (CloudManager.EC2, CloudManager.AWS):
            for name, instance in self.instances_client_aws.items():
                if name in self.index_extra_vm:
                    continue
                self.index_extra_vm.append(name)
                for loc in self.loc_aws.values():
                    if loc.region == region:
                        for zone in loc.zones:
                            logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                         region))
                            return instance, CloudManager.ON_DEMAND, loc, zone
                logging.error("<Scheduler>: Location {} not included in environment".format(region))
        elif provider.lower() in (CloudManager.GCLOUD, CloudManager.GCP):
            for name, instance in self.instances_client_gcp.items():
                if name in self.index_extra_vm:
                    continue
                self.index_extra_vm.append(name)
                for loc in self.loc_gcp.values():
                    if loc.region == region:
                        for zone in loc.zones:
                            logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                         region))
                            return instance, CloudManager.ON_DEMAND, loc, zone
                logging.error("<Scheduler>: Location {} not included in environment".format(region))
        elif provider.lower() in CloudManager.CLOUDLAB.lower():
            aux_region = 'CloudLab_' + region
            for name, instance in self.instances_server_cloudlab.items():
                if name in self.index_extra_vm:
                    continue
                if aux_region not in instance.locations:
                    continue
                self.index_extra_vm.append(name)
                for loc in self.loc_cloudlab.values():
                    if loc.region == region:
                        logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                         region))
                        return instance, Experiment.MARKET, loc, ""
                logging.error("<Scheduler>: Location {} not included in environment".format(region))

    def __separate_location_by_cloud(self, locations):
        for loc_id, loc in locations.items():
            if loc.provider in (CloudManager.EC2, CloudManager.AWS):
                self.loc_aws[loc_id] = loc
            elif loc.provider in (CloudManager.GCLOUD, CloudManager.GCP):
                self.loc_gcp[loc_id] = loc
            elif loc.provider in CloudManager.CLOUDLAB:
                self.loc_cloudlab[loc_id] = loc
            else:
                logging.error(f"<Scheduler>: {loc.provider} does not have support ({loc_id})")
