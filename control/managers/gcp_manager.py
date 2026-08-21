from control.managers.cloud_manager import CloudManager

from control.config.gcp_config import GCPConfig
from control.config.storage_config import StorageConfig

from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

from datetime import datetime
from dateutil.tz import tzutc

import logging

import iso8601

import requests

from ratelimit import limits, sleep_and_retry

from pathlib import Path

import threading

import subprocess
import sys

import json

file = open(Path(Path.home(), 'gcloud_api_key'), 'r')
api_key = file.read()


class GCPManager(CloudManager):
    gcp_config = GCPConfig()
    storage_config = StorageConfig()
    vm_config = gcp_config
    bucket_config = storage_config

    mutex = threading.Lock()

    def __init__(self):
        self.instance_client = compute_v1.InstancesClient()
        self.image_client = compute_v1.ImagesClient()
        self.disk_client = compute_v1.DisksClient()

        self.instances_history = {}

    def _update_history(self, instances, status):

        for instance in instances:
            # print("Peguei uma instância - falta terminar isso! - status: ", status)
            # with open('instance.txt', 'w') as f:
            #     print(instance, file=f)

            if status == 'start':
                self.instances_history = {
                    instance.id: {
                        'StartTime': instance.creation_timestamp,
                        'EndTime': None,
                        'Instance': instance,
                        'Zone': instance.zone
                    }
                }

            if status == 'terminate':
                if instance.id in self.instances_history:
                    self.instances_history[instance.id]['EndTime'] = \
                        datetime.now(tz=tzutc())

    def _create_instance(self, instance, zone):

        self.mutex.acquire()

        try:
            request = compute_v1.InsertInstanceRequest()
            request.zone = zone
            request.project = self.gcp_config.project
            request.instance_resource = instance

            operation = self.instance_client.insert(request=request)

            self._wait_for_operation(operation, "instance creation")

            self.mutex.release()

            instance = self.__get_instance(instance_id=instance.name, zone=zone)

            self._update_history([instance], 'start')

            return instance

        except Exception as e:
            logging.error("<GCPManager>: Error to create instance")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def create_volume(self, size, volume_name='', zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        try:
            disk = compute_v1.Disk()
            disk.type_ = f'projects/{self.gcp_config.project}/zones/{zone}/diskTypes/pd-balanced'
            disk.name = volume_name
            disk.size_gb = size

            self.mutex.acquire()

            request = compute_v1.InsertDiskRequest(
                disk_resource=disk,
                project=self.gcp_config.project,
                zone=zone,
            )

            # Make the request
            operation = self.disk_client.insert(request=request)

            self._wait_for_operation(operation, "disk insertion")

            self.mutex.release()

            disk = self.__get_disk(volume_name, zone=zone)

            return disk.name if disk else None

        except Exception as e:

            logging.error("<GCPManager>: Error to create Volume")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def wait_volume(self, volume_name='', zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        disk = self.__get_disk(volume_name, zone=zone)

        ready = False

        while disk is not None and not ready:
            if not disk.last_attach_timestamp:
                ready = True
            elif disk.last_detach_timestamp:
                last_attach_time = iso8601.parse_date(disk.last_attach_timestamp)
                last_detach_time = iso8601.parse_date(disk.last_detach_timestamp)
                ready = last_detach_time > last_attach_time
            if not ready:
                disk = self.__get_disk(volume_name, zone=zone)

    def attach_volume(self, instance_id, zone='', volume_name=''):

        if zone == '':
            zone = self.gcp_config.zone

        try:
            instance = self.__get_instance(instance_id, zone)

            disk = self.__get_disk(volume_name, zone)


            if disk is not None:
                attached_disk = compute_v1.AttachedDisk()
                attached_disk.source = disk.self_link

                request = compute_v1.AttachDiskInstanceRequest(
                    attached_disk_resource=attached_disk,
                    instance=instance.name,
                    project=self.gcp_config.project,
                    zone=zone,
                )

                self.mutex.acquire()

                operation = self.instance_client.attach_disk(request=request)
                
                self._wait_for_operation(operation, "disk attachment")

                self.mutex.release()

                return True
            else:
                return False
        except Exception as e:
            logging.error("<GCPManager>: Error to attach volume {} to instance {}".format(volume_name,
                                                                                          instance_id))
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return False

    def create_on_demand_instance(self, instance_type, image_id, zone='', vm_name='', gpu_type='', gpu_count=0):

        try:

            if zone == '':
                zone = self.gcp_config.zone
            machine_type = f'zones/{zone}/machineTypes/{instance_type}'

            self.mutex.acquire()

            # Initialize request argument(s)
            request = compute_v1.GetImageRequest(
                image=f"{image_id}",
                project=self.gcp_config.project,
            )

            # Make the request
            image = self.image_client.get(request=request)

            # Handle the response

            self.mutex.release()

            disk_type = f"zones/{zone}/diskTypes/{self.gcp_config.disk_type}"
            boot_disk = compute_v1.AttachedDisk()
            initialize_params = compute_v1.AttachedDiskInitializeParams()
            initialize_params.source_image = image.self_link
            # initialize_params.disk_type = disk_type
            boot_disk.initialize_params = initialize_params
            boot_disk.auto_delete = True
            boot_disk.boot = True
            disks = [boot_disk, ]

            network_interface = compute_v1.NetworkInterface()
            network_interface.network = f"global/networks/{self.gcp_config.network}"

            access = compute_v1.AccessConfig()
            access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
            access.name = "External NAT"
            access.network_tier = access.NetworkTier.PREMIUM.name
            network_interface.access_configs = [access]

            # Collect information into the Instance object.
            instance = compute_v1.Instance()
            instance.network_interfaces = [network_interface]
            instance.name = vm_name
            instance.disks = disks
            instance.machine_type = machine_type

            instance.scheduling = compute_v1.Scheduling()

            instance.metadata = compute_v1.Metadata({
                "items": [
                    {
                        "key": 'enable-osconfig',
                        "value": 'TRUE'
                    },
                    {
                        "key": 'enable-oslogin',
                        "value": 'true'
                    }
                ]
            })

            
            instance.tags =  compute_v1.Tags({'items': ['http-server', 'https-server', 'all-in', 'all-out']})

            if gpu_count > 0:
                logging.error("Not tested with GPU yet!")
                # config = {
                #     'name': vm_name,
                #     'machineType': machine_type,

                #     # Not working. Still in Beta on GCP API!
                #     # # 'sourceMachineImage': f'projects/{self.gcp_config.project}/machineImages/{image_id}',
                #     # 'sourceMachineImage': source_machine_image,

                #     # Specify the boot disk and the image to use as a source.
                #     'disks': [
                #         {
                #             'boot': True,
                #             'autoDelete': True,
                #             'initializeParams': {
                #                 'sourceImage': source_disk_image,
                #             }

                #         }
                #     ],

                #     # Allowing SSH connection from third-parties
                #     "metadata": {
                #         "items": [
                #             {
                #                 "key": 'enable-oslogin',
                #                 "value": 'TRUE'
                #             }
                #         ]
                #     },

                #     # Allow the instance to access cloud storage.
                #     'serviceAccounts': [{
                #         'email': 'default',
                #         'scopes': [
                #             'https://www.googleapis.com/auth/devstorage.read_write'
                #         ]
                #     }],

                #     "guestAccelerators":
                #     [
                #         {
                #             "acceleratorCount": gpu_count,
                #             "acceleratorType": f"projects/{self.gcp_config.project}/zones/{zone}/"
                #                                f"acceleratorTypes/{gpu_type}"
                #         }
                #     ],

                #     # Specify a network interface with NAT to access the public
                #     # internet.
                #     'networkInterfaces': [{
                #         'network': 'global/networks/default',
                #         'accessConfigs': [
                #             {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                #         ]
                #     }],
                #     'tags': [{
                #         'items': ['http-server', 'https-server']
                #     }],
                #     "scheduling":
                #     {
                #         "onHostMaintenance": "terminate"
                #     }
                # }

            instance = self._create_instance(instance, zone)

            if instance is not None:
                return instance.name
            else:
                return None

        except Exception as e:
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def delete_volume(self, volume_id, zone='', volume_name=''):
        logging.error("<GCPManager> delete_volume not updated yet!")
        if zone == '':
            zone = self.gcp_config.zone
        try:
            self.mutex.acquire()
            self.compute_engine.disks().delete(project=self.gcp_config.project, zone=zone,
                                               disk=volume_name).execute()
            self.mutex.release()
            status = True
        except Exception as e:
            logging.error("<GCPManager>: Error to delete Volume {} ({}) ".format(volume_id, volume_name))
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            status = False

        return status

    def create_preemptible_instance(self, instance_type, image_id, zone='', vm_name='', gpu_type='', gpu_count=0):

        try:

            if zone == '':
                zone = self.gcp_config.zone
            machine_type = f'zones/{zone}/machineTypes/{instance_type}'

            self.mutex.acquire()

            # Initialize request argument(s)
            request = compute_v1.GetImageRequest(
                image=f"{image_id}",
                project=self.gcp_config.project,
            )

            # Make the request
            image = self.image_client.get(request=request)

            # Handle the response
            self.mutex.release()

            disk_type = f"zones/{zone}/diskTypes/{self.gcp_config.disk_type}"
            boot_disk = compute_v1.AttachedDisk()
            initialize_params = compute_v1.AttachedDiskInitializeParams()
            initialize_params.source_image = image.self_link
            # initialize_params.disk_type = disk_type
            boot_disk.initialize_params = initialize_params
            boot_disk.auto_delete = True
            boot_disk.boot = True
            disks = [boot_disk, ]

            network_interface = compute_v1.NetworkInterface()
            network_interface.network = f"global/networks/{self.gcp_config.network}"

            access = compute_v1.AccessConfig()
            access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
            access.name = "External NAT"
            access.network_tier = access.NetworkTier.PREMIUM.name
            network_interface.access_configs = [access]

            # Collect information into the Instance object.
            instance = compute_v1.Instance()
            instance.network_interfaces = [network_interface]
            instance.name = vm_name
            instance.disks = disks
            instance.machine_type = machine_type

            instance.scheduling = compute_v1.Scheduling()
            instance.scheduling.provisioning_model = "SPOT"

            instance.metadata = compute_v1.Metadata({
                "items": [
                    {
                        "key": 'enable-osconfig',
                        "value": 'TRUE'
                    },
                    {
                        "key": 'enable-oslogin',
                        "value": 'true'
                    }
                ]
            })

            
            instance.tags =  compute_v1.Tags({'items': ['http-server', 'https-server', 'all-in', 'all-out']})

            if gpu_count > 0:
                logging.error("Not tested with GPU yet!")
                # # print("creating with GPU")
                # config = {
                #     'name': vm_name,
                #     'machineType': machine_type,

                #     # Not working. Still in Beta on GCP API!
                #     # # 'sourceMachineImage': f'projects/{self.gcp_config.project}/machineImages/{image_id}',
                #     # 'sourceMachineImage': source_machine_image,

                #     # Specify the boot disk and the image to use as a source.
                #     'disks': [
                #         {
                #             'boot': True,
                #             'autoDelete': True,
                #             'initializeParams': {
                #                 'sourceImage': source_disk_image,
                #             }

                #         }
                #     ],

                #     # Allowing SSH connection from third-parties
                #     "metadata": {
                #         "items": [
                #             {
                #                 "key": 'enable-oslogin',
                #                 "value": 'TRUE'
                #             }
                #         ]
                #     },

                #     # Allow the instance to access cloud storage.
                #     'serviceAccounts': [{
                #         'email': 'default',
                #         'scopes': [
                #             'https://www.googleapis.com/auth/devstorage.read_write'
                #         ]
                #     }],

                #     "guestAccelerators":
                #     [
                #         {
                #             "acceleratorCount": gpu_count,
                #             "acceleratorType": f"projects/{self.gcp_config.project}/zones/{zone}/"
                #                                f"acceleratorTypes/{gpu_type}"
                #         }
                #     ],

                #     # Specify a network interface with NAT to access the public
                #     # internet.
                #     'networkInterfaces': [{
                #         'network': 'global/networks/default',
                #         'accessConfigs': [
                #             {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                #         ]
                #     }],
                #     'tags': [{
                #         'items': ['http-server', 'https-server']
                #     }],
                #     "scheduling":
                #     {
                #         "onHostMaintenance": "terminate",
                #         "provisioningModel": "SPOT"
                #     }
                # }

            instance = self._create_instance(instance, zone)

            if instance is not None:
                return instance.name
            else:
                return None

        except Exception as e:
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def _terminate_instance(self, instance, zone):

        try:

            self._update_history([instance], status='terminate')

            self.mutex.acquire()

            request = compute_v1.DeleteInstanceRequest(
                instance=instance.name,
                project=self.gcp_config.project,
                zone=zone,
            )

            operation = self.instance_client.delete(request=request)

            self.mutex.release()

            return operation

        except Exception as e:
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def terminate_instance(self, instance_id, wait=True, zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        try:
            instance = self.__get_instance(instance_id, zone)
            operation = self._terminate_instance(instance, zone=zone)

            if wait:
                self.mutex.acquire()
                self._wait_for_operation(operation, "instance deletion")
                self.mutex.release()

            status = True

        except Exception as e:
            logging.error("<GCPManager>: Error to terminate instance {}".format(instance_id))
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()

            status = False

        return status

    @sleep_and_retry
    @limits(calls=10, period=1)
    def __get_instance(self, instance_id, zone):
        try:

            instance = self.instance_client.get(project=self.gcp_config.project, zone=zone, instance=instance_id)

        except Exception as e:
            logging.info(e)
            return None

        return instance

    def get_instance_status(self, instance_id, zone=''):
        if instance_id is None:
            return None

        if zone == '':
            zone = self.gcp_config.zone

        instance = self.__get_instance(instance_id=instance_id, zone=zone)

        if instance is None:
            print("instance status", CloudManager.TERMINATED)
            return CloudManager.TERMINATED
        else:
            # print("instance status", instance.status)

            return instance.status.lower()

    def __get_disk(self, disk_name, zone):

        try:
            request = compute_v1.GetDiskRequest(
                disk=disk_name,
                project=self.gcp_config.project,
                zone=zone,
            )
            self.mutex.acquire()
            ret = self.disk_client.get(request=request)
            self.mutex.release()
            return ret
        except Exception as e:
            logging.error("<GCPManager>: Error to find disk")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def get_public_instance_ip(self, instance_id, zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        instance = self.__get_instance(instance_id=instance_id, zone=zone)
        if instance is None:
            return None
        else:
            return instance.network_interfaces[0].access_configs[0].nat_i_p

    def get_private_instance_ip(self, instance_id, zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        instance = self.__get_instance(instance_id=instance_id, zone=zone)
        if instance is None:
            return None
        else:
            return instance.network_interfaces[0].network_i_p

    @staticmethod
    def get_preemptible_price(instance_type, region=None):
        params = {'pageToken': None,
                  'filter': 'service="services/6F81-5844-456A"'}

        sku_data_gcp = []
        instance_data_gcp = []

        termina = False

        while len(sku_data_gcp) < 2 and not termina:

            if params['pageToken'] is None:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v2beta/skus?key={api_key}&pageSize=5000',
                    params={'filter': 'service="services/6F81-5844-456A"'})
            else:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v2beta/skus?key={api_key}&pageSize=5000',
                    params=params)

            all_data_gcp = r.json()

            try:
                params['pageToken'] = all_data_gcp['nextPageToken']
            except Exception as e:
                params['pageToken'] = None
                termina = True

            aux_list_2 = [x for x in all_data_gcp['skus']
                        if (x['displayName'].startswith(f'Spot Preemptible {instance_type.split("-")[0].upper()} Instance Core')
                            or 
                            x['displayName'].startswith(f'Spot Preemptible {instance_type.split("-")[0].upper()} Instance Ram'))
                        ]

            aux_list = []
            
            for x in aux_list_2:
                if x['geoTaxonomy']['type'] == 'TYPE_MULTI_REGIONAL':
                    for reg in x['geoTaxonomy']['multiRegionalMetadata']['regions']:
                        if region == reg['region']:
                            aux_list.append(x)
                            break
                if (x['geoTaxonomy']['type'] == 'TYPE_REGIONAL' and region in x['geoTaxonomy']['regionalMetadata']['region']):
                    aux_list.append(x)

            for a in aux_list:
                sku_data_gcp.append(a)

        # print(json.dumps(sku_data_gcp, indent=2))

        for sku in sku_data_gcp:
            sku_id = sku["skuId"]
            r = requests.get(
                url=f'https://cloudbilling.googleapis.com/v1beta/skus/{sku_id}/price?key={api_key}')
                
            sku_data = r.json()

            instance_data_gcp.append(sku_data)

        # print(json.dumps(instance_data_gcp, indent=2))

        if 'Core' in sku_data_gcp[0]['displayName']:
            try:
                int_price_per_vcpu = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_vcpu = 0
            cents_per_vcpu = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            try:
                int_price_per_ram = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_ram = 0
            cents_per_ram = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram
        else:
            try:
                int_price_per_vcpu = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_vcpu = 0
            cents_per_vcpu = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            try:
                int_price_per_ram = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_ram = 0
            cents_per_ram = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram

        # print("price_per_vcpu: ", price_per_vcpu)
        # print("price_per_ram: ", price_per_ram)
        return price_per_vcpu, price_per_ram

    # Get current GCP price for an on-demand instance
    @staticmethod
    def get_ondemand_price(instance_type, region):

        params = {'pageToken': None,
                  'filter': 'service="services/6F81-5844-456A"'}

        sku_data_gcp = []
        instance_data_gcp = []

        # print("instance type:'", instance_type, "'")
        # print("region: ", region)

        termina = False

        while len(sku_data_gcp) < 2 and not termina:

            if params['pageToken'] is None:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v2beta/skus?key={api_key}&pageSize=5000',
                    params={'filter': 'service="services/6F81-5844-456A"'})
            else:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v2beta/skus?key={api_key}&pageSize=5000',
                    params=params)

            all_data_gcp = r.json()

            try:
                params['pageToken'] = all_data_gcp['nextPageToken']
            except Exception as e:
                params['pageToken'] = None
                termina = True

            aux_list_2 = [x for x in all_data_gcp['skus']
                        if (x['displayName'].startswith(f'{instance_type.split("-")[0].upper()} Instance Core')
                            or 
                            x['displayName'].startswith(f'{instance_type.split("-")[0].upper()} Instance Ram'))
                        ]

            aux_list = []

            for x in aux_list_2:
                if x['geoTaxonomy']['type'] == 'TYPE_MULTI_REGIONAL':
                    for reg in x['geoTaxonomy']['multiRegionalMetadata']['regions']:
                        if region == reg['region']:
                            aux_list.append(x)
                            break
                if (x['geoTaxonomy']['type'] == 'TYPE_REGIONAL' and region in x['geoTaxonomy']['regionalMetadata']['region']):
                    aux_list.append(x)

            for a in aux_list:
                sku_data_gcp.append(a)

        # print(json.dumps(sku_data_gcp, indent=2))

        for sku in sku_data_gcp:
            sku_id = sku["skuId"]
            r = requests.get(
                url=f'https://cloudbilling.googleapis.com/v1beta/skus/{sku_id}/price?key={api_key}')
                
            sku_data = r.json()

            instance_data_gcp.append(sku_data)

        # print(json.dumps(instance_data_gcp, indent=2))

        if 'Core' in sku_data_gcp[0]['displayName']:
            try:
                int_price_per_vcpu = int(instance_data_gcp[0]['rate']['tiers']
                                         [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_vcpu = 0
            cents_per_vcpu = int(instance_data_gcp[0]['rate']['tiers']
                                         [0]['listPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            try:
                int_price_per_ram = int(instance_data_gcp[1]['rate']['tiers']
                                         [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_ram = 0
            cents_per_ram = int(instance_data_gcp[1]['rate']['tiers']
                                         [0]['listPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram
        else:
            try:
                int_price_per_vcpu = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_vcpu = 0
            cents_per_vcpu = int(instance_data_gcp[1]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            try:
                int_price_per_ram = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['units'])
            except Exception as e:
                int_price_per_ram = 0
            cents_per_ram = int(instance_data_gcp[0]['rate']['tiers']
                                            [0]['listPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram

        # print("price_per_vcpu: ", price_per_vcpu)
        # print("price_per_ram: ", price_per_ram)
        return price_per_vcpu, price_per_ram

    # Get current on-demand GPU price for GCP
    @staticmethod
    def get_ondemand_gpu_price(gpu_type, region):

        params = {'pageToken': None}

        gpu_data_gcp = []

        gpu_type_search = gpu_type.replace("-", " ")

        # print("gpu type:'", gpu_type, "'")
        # print("gpu type search:'", gpu_type_search, "'")
        # print("region: ", region)

        while len(gpu_data_gcp) < 1:

            if params['pageToken'] is None:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?key={api_key}')
            else:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?key={api_key}',
                    params=params)

            all_data_gcp = r.json()

            params['pageToken'] = all_data_gcp['nextPageToken']

            aux_list = [x for x in all_data_gcp['skus']
                        if gpu_type_search.lower() in x['description'].lower()
                        and 'Preemptible' not in x['description']
                        and 'OnDemand' in x['category']['usageType']]

            aux_list = [x for x in aux_list if region in x['serviceRegions']]

            for a in aux_list:
                gpu_data_gcp.append(a)

        int_price_per_gpu = int(gpu_data_gcp[0]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates']
                                [0]['unitPrice']['units'])
        cents_per_gpu = int(gpu_data_gcp[0]['pricingInfo'][0]
                            ['pricingExpression']['tieredRates'][0]
                            ['unitPrice']['nanos']) / 1000000000
        price_per_gpu = int_price_per_gpu + cents_per_gpu

        # print("price_per_gpu: ", price_per_gpu)
        return price_per_gpu

    # Get current on-demand GPU price for GCP
    @staticmethod
    def get_preemptible_gpu_price(gpu_type, region):

        params = {'pageToken': None}

        gpu_data_gcp = []

        gpu_type_search = gpu_type.replace("-", " ")

        # print("gpu type:'", gpu_type, "'")
        # print("gpu type search:'", gpu_type_search, "'")
        # print("region: ", region)

        while len(gpu_data_gcp) < 1:

            if params['pageToken'] is None:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?key={api_key}')
            else:
                r = requests.get(
                    url=f'https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?key={api_key}',
                    params=params)

            all_data_gcp = r.json()

            params['pageToken'] = all_data_gcp['nextPageToken']

            aux_list = [x for x in all_data_gcp['skus']
                        if gpu_type_search.lower() in x['description'].lower()
                        and 'Preemptible' in x['description']]

            aux_list = [x for x in aux_list if region in x['serviceRegions']]

            for a in aux_list:
                gpu_data_gcp.append(a)

        int_price_per_gpu = int(gpu_data_gcp[0]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates']
                                [0]['unitPrice']['units'])
        cents_per_gpu = int(gpu_data_gcp[0]['pricingInfo'][0]
                            ['pricingExpression']['tieredRates'][0]
                            ['unitPrice']['nanos']) / 1000000000
        price_per_gpu = int_price_per_gpu + cents_per_gpu

        # print("price_per_gpu: ", price_per_gpu)
        return price_per_gpu

    # Get availability zones of a GCP region
    def get_availability_zones(self, region):
        # Get zones info
        client = compute_v1.RegionZonesClient()
        
        # Initialize request argument(s)
        request = compute_v1.ListRegionZonesRequest(
            project=self.gcp_config.project,
            region=region,
        )

        # Make the request
        page_result = client.list(request=request)

        zones = []

        # Handle the response
        for response in page_result:
            zones.append(response.name)
        
        zones.sort()

        return zones

    def _wait_for_operation(self, operation: ExtendedOperation, verbose_name: str = "operation", timeout: int = 300):
        # print('Waiting for operation to finish...')

        result = operation.result(timeout=timeout)

        if operation.error_code:
            logging.error(
                f"Error during {verbose_name}: [Code: {operation.error_code}]: {operation.error_message}",
                file=sys.stderr,
                flush=True,
            )
            logging.error(f"Operation ID: {operation.name}", file=sys.stderr, flush=True)
            raise operation.exception() or RuntimeError(operation.error_message)

        if operation.warnings:
            logging.warning(f"Warnings during {verbose_name}:\n", file=sys.stderr, flush=True)
            for warning in operation.warnings:
                logging.warning(f" - {warning.code}: {warning.message}", file=sys.stderr, flush=True)

        return result

    def execute_command(self, instance_name, command, zone='', simple_quotes=False):
        if zone == '':
            zone = self.gcp_config.zone
        logging.info(f"<GCPManager> Executing command {command} in instance {instance_name}")
        if simple_quotes:
            # print(f"gcloud compute ssh --zone {zone} --project {self.gcp_config.project} {instance_name} --command=\'{command}\'")
            return subprocess.run(f"gcloud compute ssh --zone {zone} --project {self.gcp_config.project} {instance_name} --command=\'{command}\' --quiet", 
                                   shell=True, check=True, capture_output=True)
        else:    
            # print(f"gcloud compute ssh --zone {zone} --project {self.gcp_config.project} {instance_name} --command=\"{command}\"")
            return subprocess.run(f"gcloud compute ssh --zone {zone} --project {self.gcp_config.project} {instance_name} --command=\"{command}\" --quiet", 
                        shell=True, check=True, capture_output=True)
    
    def send_file(self, instance_name, source, target, zone=''):
        if zone == '':
            zone = self.gcp_config.zone
        logging.info(f"<GCPManager> Sending file {source} to {target} in instance {instance_name}")
        # print(f"gcloud compute scp {source} --zone {zone} --project {self.gcp_config.project} {instance_name}:{target}")
        subprocess.run(f"gcloud compute scp {source} --zone {zone} --project {self.gcp_config.project} {instance_name}:{target}", 
                       shell=True, check=True, capture_output=True)
        
    def restart_instance(self, instance_id, zone=''):
        if zone == '':
            zone = self.gcp_config.zone

        self.mutex.acquire()

        try:
            instance = self.__get_instance(instance_id, zone)
            request = compute_v1.StartInstanceRequest(
                instance=instance.name,
                zone=zone,
                project=self.gcp_config.project,
            )

            self.instance_client.start(request=request)

            self.mutex.release()

        except Exception as e:
            logging.error("<GCPManager>: Error to create instance")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()

    def download_file(self, instance_name, source, target, zone=''):
            if zone == '':
                zone = self.gcp_config.zone
            logging.info(f"<GCPManager> Sending file {source} to {target} in instance {instance_name}")
            # print(f"gcloud compute scp {source} --zone {zone} --project {self.gcp_config.project} {instance_name}:{target}")
            subprocess.run(f"gcloud compute scp {instance_name}:{source} --zone {zone} --project {self.gcp_config.project} {target}", 
                           shell=True, check=True, capture_output=True)