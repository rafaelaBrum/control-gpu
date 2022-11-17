from control.domain.instance_type import InstanceType
from control.domain.cloud_region import CloudRegion

from control.managers.cloud_manager import CloudManager

from typing import Dict

import logging


class Scheduler:
    FL_SIMPLE = "SIMPLE"
    MAT_FORM = "FORMULATION"

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

    def get_server_initial_instance(self, provider, region, vm_name):
        logging.info("<Scheduler>: Choosing initial instance for server task from provider {}".format(provider))
        if provider.lower() in (CloudManager.EC2, CloudManager.AWS):
            for name, instance in self.instances_server_aws.items():
                if name == vm_name:
                    for loc in self.loc_aws.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                return instance, CloudManager.ON_DEMAND, loc, zone
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
                                return instance, CloudManager.ON_DEMAND, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))

    def get_client_initial_instance(self, provider, region, vm_name):
        logging.info("<Scheduler>: Choosing initial instance for client task from provider {}".format(provider))
        if provider.lower() in (CloudManager.EC2, CloudManager.AWS):
            for name, instance in self.instances_client_aws.items():
                if name == vm_name:
                    for loc in self.loc_aws.values():
                        if loc.region == region:
                            for zone in loc.zones:
                                logging.info("<Scheduler>: On-demand instance chosen {} in region {}".format(name,
                                                                                                             region))
                                return instance, CloudManager.ON_DEMAND, loc, zone
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
                                return instance, CloudManager.ON_DEMAND, loc, zone
                    logging.error("<Scheduler>: Location {} not included in environment".format(region))
            logging.error("<Scheduler>: Instance {} not included in environment".format(vm_name))

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
