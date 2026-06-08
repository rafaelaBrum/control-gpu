from control.managers.cloud_manager import CloudManager

from control.config.gcp_config import GCPConfig
from control.config.storage_config import StorageConfig

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

from datetime import datetime
from dateutil.tz import tzutc
# from datetime import timedelta

import logging
# import json

# import math
import time
import iso8601

import requests

from ratelimit import limits, sleep_and_retry

from pathlib import Path

import threading


# TODO: get API key from GClod Python API
file = open(Path(Path.home(), 'gcloud_api_key'), 'r')
api_key = file.read()


class GCPManager(CloudManager):
    gcp_conf = GCPConfig()
    storage_config = StorageConfig()
    vm_config = gcp_conf
    bucket_config = storage_config

    mutex = threading.Lock()
    credentials = GoogleCredentials.get_application_default()
    compute_engine = googleapiclient.discovery.build('compute', 'v1',
                                                     credentials=credentials, cache_discovery=True)

    def __init__(self):

        self.instances_history = {}

    def _update_history(self, instances, status):

        for instance in instances:

            if status == 'start':
                self.instances_history = {
                    instance['id']: {
                        'StartTime': instance['creationTimestamp'],
                        'EndTime': None,
                        'Instance': instance,
                        'Zone': instance['zone']
                    }
                }

            if status == 'terminate':
                if instance['id'] in self.instances_history:
                    self.instances_history[instance['id']]['EndTime'] = \
                        datetime.now(tz=tzutc())

    def _wait_for_operation(self, operation, zone):
        # print('Waiting for operation to finish...')

        while True:
            result = self.compute_engine.zoneOperations().get(
                project=self.gcp_conf.project,
                zone=zone,
                operation=operation).execute()

            if result['status'] == 'DONE':
                # print("done.")
                if 'error' in result:
                    raise Exception(result['error'])
                return result

            time.sleep(1)

    def _create_instance(self, info, zone):

        self.mutex.acquire()

        try:
            operation = self.compute_engine.instances().insert(
                project=self.gcp_conf.project,
                zone=zone,
                body=info).execute()

            self._wait_for_operation(operation['name'], zone=zone)

            self.mutex.release()

            instances = self.__get_instances(get_filter=f'(name = {info["name"]})', zone=zone)

            self._update_history(instances, 'start')

            return instances

        except Exception as e:
            logging.error("<GCPManager>: Error to create instance")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def create_volume(self, size, volume_name='', zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        try:
            disk_body = {
                'name': volume_name,
                "sizeGb": size,
                'type': f'projects/{self.gcp_conf.project}/zones/{zone}/diskTypes/pd-balanced'
            }

            self.mutex.acquire()

            operation = self.compute_engine.disks().insert(project=self.gcp_conf.project, zone=zone,
                                                           body=disk_body).execute()

            self._wait_for_operation(operation['name'], zone=zone)

            self.mutex.release()

            disk = self.__get_disk(volume_name, zone=zone)

            return disk['name'] if disk else None

        except Exception as e:

            logging.error("<GCPManager>: Error to create Volume")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def wait_volume(self, volume_name='', zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        disk = self.__get_disk(volume_name, zone=zone)

        ready = False

        while disk is not None and not ready:
            if 'lastAttachTimestamp' not in disk:
                ready = True
            elif 'lastDetachTimestamp' in disk:
                last_attach_time = iso8601.parse_date(disk['lastAttachTimestamp'])
                last_detach_time = iso8601.parse_date(disk['lastDetachTimestamp'])
                ready = last_detach_time > last_attach_time
            if not ready:
                disk = self.__get_disk(volume_name, zone=zone)

    def attach_volume(self, instance_id, zone='', volume_name=''):

        if zone == '':
            zone = self.gcp_conf.zone

        try:
            instance = self.__get_instance(instance_id, zone)

            self.mutex.acquire()

            disk = self.compute_engine.disks().get(project=self.gcp_conf.project, zone=zone,
                                                   disk=volume_name).execute()

            self.mutex.release()

            if disk is not None:
                attached_disk_body = {
                    'source': disk['selfLink']
                }

                self.mutex.acquire()

                operation = self.compute_engine.instances().attachDisk(project=self.gcp_conf.project,
                                                                       zone=zone,
                                                                       instance=instance['name'],
                                                                       body=attached_disk_body).execute()

                self._wait_for_operation(operation['name'], zone=zone)

                self.mutex.release()

                return True
            else:
                return False
        except Exception as e:
            logging.error("<GCPManager>: Error to attach volume {} ({}} to instance {}".format(volume_id,
                                                                                               volume_name,
                                                                                               instance_id))
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return False

    def create_on_demand_instance(self, instance_type, image_id, zone='', vm_name='', gpu_type='', gpu_count=0):

        try:

            if zone == '':
                zone = self.gcp_conf.zone
            machine_type = f'zones/{zone}/machineTypes/{instance_type}'

            self.mutex.acquire()

            image_response = self.compute_engine.images().get(project=self.gcp_conf.project,
                                                              image=f'{image_id}').execute()
            self.mutex.release()

            source_disk_image = image_response['selfLink']

            if gpu_count > 0:
                # print("creating with GPU")
                config = {
                    'name': vm_name,
                    'machineType': machine_type,

                    # Not working. Still in Beta on GCP API!
                    # # 'sourceMachineImage': f'projects/{self.gcp_conf.project}/machineImages/{image_id}',
                    # 'sourceMachineImage': source_machine_image,

                    # Specify the boot disk and the image to use as a source.
                    'disks': [
                        {
                            'boot': True,
                            'autoDelete': True,
                            'initializeParams': {
                                'sourceImage': source_disk_image,
                            }

                        }
                    ],

                    # Allowing SSH connection from third-parties
                    "metadata": {
                        "items": [
                            {
                                "key": 'enable-oslogin',
                                "value": 'TRUE'
                            }
                        ]
                    },

                    # Allow the instance to access cloud storage.
                    'serviceAccounts': [{
                        'email': 'default',
                        'scopes': [
                            'https://www.googleapis.com/auth/devstorage.read_write'
                        ]
                    }],

                    "guestAccelerators":
                    [
                        {
                            "acceleratorCount": gpu_count,
                            "acceleratorType": f"projects/{self.gcp_conf.project}/zones/{zone}/"
                                               f"acceleratorTypes/{gpu_type}"
                        }
                    ],

                    # Specify a network interface with NAT to access the public
                    # internet.
                    'networkInterfaces': [{
                        'network': 'global/networks/default',
                        'accessConfigs': [
                            {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                        ]
                    }],
                    'tags': [{
                        'items': ['http-server', 'https-server']
                    }],
                    "scheduling":
                    {
                        "onHostMaintenance": "terminate"
                    }
                }
            else:
                config = {
                    'name': vm_name,
                    'machineType': machine_type,

                    # Not working. Still in Beta on GCP API!
                    # # 'sourceMachineImage': f'projects/{self.gcp_conf.project}/machineImages/{image_id}',
                    # 'sourceMachineImage': source_machine_image,

                    # Specify the boot disk and the image to use as a source.
                    'disks': [
                        {
                            'boot': True,
                            'autoDelete': True,
                            'initializeParams': {
                                'sourceImage': source_disk_image,
                            }

                        }
                    ],

                    # Allowing SSH connection from third-parties
                    "metadata": {
                        "items": [
                            {
                                "key": 'enable-oslogin',
                                "value": 'TRUE'
                            }
                        ]
                    },

                    # Allow the instance to access cloud storage.
                    'serviceAccounts': [{
                        'email': 'default',
                        'scopes': [
                            'https://www.googleapis.com/auth/devstorage.read_write'
                        ]
                    }],

                    # Specify a network interface with NAT to access the public
                    # internet.
                    'networkInterfaces': [{
                        'network': 'global/networks/default',
                        'accessConfigs': [
                            {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                        ]
                    }],
                    'tags': [{
                        'items': ['http-server', 'https-server']
                    }]
                }

            instances = self._create_instance(config, zone)

            if instances is not None:
                created_instances = [i for i in instances]
                instance = created_instances[0]
                return instance['id']
            else:
                return None

        except Exception as e:
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def delete_volume(self, volume_id, zone='', volume_name=''):
        if zone == '':
            zone = self.gcp_conf.zone
        try:
            self.mutex.acquire()
            self.compute_engine.disks().delete(project=self.gcp_conf.project, zone=zone,
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

    def create_preemptible_instance(self, instance_type, image_id, max_price, zone=''):
        pass

        # user_data = '''#!/bin/bash
        # /usr/bin/enable-ec2-spot-hibernation
        # echo "user-data" > $HOME/control-applications/test_user_data.txt
        # '''

        # zone = self.ec2_conf.zone
        # interruption_behaviour = 'stop'
        #
        # parameters = {
        #     'ImageId': image_id,
        #     'InstanceType': instance_type,
        #     'KeyName': self.ec2_conf.key_name,
        #     'MaxCount': 1,
        #     'MinCount': 1,
        #     'SecurityGroups': [
        #         self.ec2_conf.security_group,
        #         self.ec2_conf.security_vpc_group
        #     ],
        #     'InstanceMarketOptions':
        #         {
        #             'MarketType': 'spot',
        #             'SpotOptions': {
        #                 'MaxPrice': str(max_price),
        #                 'SpotInstanceType': 'persistent',
        #                 'InstanceInterruptionBehavior': interruption_behaviour
        #             }
        #         },
        #     'Placement': {}
        #     # 'UserData': user_data
        # }
        #
        # if zone:
        #     parameters['Placement'] = {'AvailabilityZone': zone}
        #
        # instances = self._create_instance(parameters, burstable)
        #
        # if instances is not None:
        #     created_instances = [i for i in instances]
        #     instance = created_instances[0]
        #     return instance.id
        # else:
        #     return None

    def _terminate_instance(self, instance, zone):
        # if instance is spot, we have to remove its request
        # if instance.instance_lifecycle == 'spot':
        #     self.client.cancel_spot_instance_requests(
        #         SpotInstanceRequestIds=[
        #             instance.spot_instance_request_id
        #         ]
        #     )

        try:

            self._update_history([instance], status='terminate')

            self.mutex.acquire()

            operation = self.compute_engine.instances().delete(project=self.gcp_conf.project,
                                                               zone=zone,
                                                               instance=instance['name']).execute()

            self.mutex.release()

            return operation

        except Exception as e:
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def terminate_instance(self, instance_id, wait=True, zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        try:
            instance = self.__get_instance(instance_id, zone)
            operation = self._terminate_instance(instance, zone=zone)

            if wait:
                self.mutex.acquire()
                self._wait_for_operation(operation['name'], zone=zone)
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

            instances = self.__get_instances(get_filter=f'(id = {instance_id})', zone=zone)

            if instances is not None:
                instance = instances[0]
            else:
                instance = None

        except Exception as e:
            logging.info(e)
            return None

        return instance

    def __get_instances(self, get_filter=None, zone=''):

        try:

            if zone == '':
                zone = self.gcp_conf.zone

            self.mutex.acquire()

            if get_filter is None:
                result = self.compute_engine.instances().list(project=self.gcp_conf.project,
                                                              zone=zone).execute()
            else:
                result = self.compute_engine.instances().list(project=self.gcp_conf.project,
                                                              zone=zone,
                                                              filter=get_filter).execute()

            self.mutex.release()
            return result['items'] if result is not None and 'items' in result else None

        except Exception as e:
            logging.info(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def get_instance_status(self, instance_id, zone=''):
        if instance_id is None:
            return None

        if zone == '':
            zone = self.gcp_conf.zone

        instances = self.__get_instances(get_filter=f'(id = {instance_id})', zone=zone)

        if instances is not None:
            instance = instances[0]
        else:
            instance = None

        if instance is None:
            # print("instance status", CloudManager.TERMINATED)
            return CloudManager.TERMINATED
        else:
            # print("instance status", instance['status'])

            return instance['status'].lower()

    def __get_disk(self, disk_name, zone):

        try:
            self.mutex.acquire()
            ret = self.compute_engine.disks().get(project=self.gcp_conf.project,
                                                  zone=zone, disk=disk_name).execute()
            self.mutex.release()
            return ret
        except Exception as e:
            logging.error("<GCPManager>: Error to find instance")
            logging.error(e)
            if self.mutex.locked():
                self.mutex.release()
            return None

    def list_instances_id(self, list_filter=None, zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        instances = self.__get_instances(list_filter, zone=zone)

        return [i['id'] for i in instances] if instances else []

    def get_public_instance_ip(self, instance_id, zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        instances = self.__get_instances(get_filter=f'(id = {instance_id})', zone=zone)
        if instances is not None:
            instance = instances[0]
        else:
            instance = None
        if instance is None:
            return None
        else:
            return instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']

    def get_private_instance_ip(self, instance_id, zone=''):
        if zone == '':
            zone = self.gcp_conf.zone
        instances = self.__get_instances(get_filter=f'(id = {instance_id})', zone=zone)
        if instances is not None:
            instance = instances[0]
        else:
            instance = None
        if instance is None:
            return None
        else:
            return instance['networkInterfaces'][0]['networkIP']

    @staticmethod
    def get_preemptible_price(instance_type, region=None):
        params = {'pageToken': None}

        instance_data_gcp = []

        while len(instance_data_gcp) < 2:

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
                        if (f'{instance_type.split("-")[0].upper()} ' in x['description']
                            and 'Instance' in x['description'])
                        and 'Preemptible' in x['description']
                        and 'Custom' not in x['description']]

            aux_list = [x for x in aux_list if region in x['serviceRegions']]

            for a in aux_list:
                instance_data_gcp.append(a)

        if 'Core' in instance_data_gcp[0]['description']:
            int_price_per_vcpu = int(instance_data_gcp[0]['pricingInfo'][0]
                                     ['pricingExpression']['tieredRates']
                                     [0]['unitPrice']['units'])
            cents_per_vcpu = int(instance_data_gcp[0]['pricingInfo'][0]
                                 ['pricingExpression']['tieredRates'][0]
                                 ['unitPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            int_price_per_ram = int(instance_data_gcp[1]['pricingInfo'][0]
                                    ['pricingExpression']['tieredRates'][0]
                                    ['unitPrice']['units'])
            cents_per_ram = int(instance_data_gcp[1]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates'][0]
                                ['unitPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram
        else:
            int_price_per_vcpu = int(instance_data_gcp[1]['pricingInfo'][0]
                                     ['pricingExpression']['tieredRates']
                                     [0]['unitPrice']['units'])
            cents_per_vcpu = int(instance_data_gcp[1]['pricingInfo'][0]
                                 ['pricingExpression']['tieredRates'][0]
                                 ['unitPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            int_price_per_ram = int(instance_data_gcp[0]['pricingInfo'][0]
                                    ['pricingExpression']['tieredRates'][0]
                                    ['unitPrice']['units'])
            cents_per_ram = int(instance_data_gcp[0]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates'][0]
                                ['unitPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram

        return price_per_vcpu, price_per_ram

    # Get current GCP price for an on-demand instance
    @staticmethod
    def get_ondemand_price(instance_type, region):

        params = {'pageToken': None}

        instance_data_gcp = []

        # print("instance type:'", instance_type, "'")
        # print("region: ", region)

        while len(instance_data_gcp) < 2:

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
                        if (f'{instance_type.split("-")[0].upper()} ' in x['description']
                            and 'Instance' in x['description'])
                        and 'Preemptible' not in x['description']
                        and 'Custom' not in x['description']]

            aux_list = [x for x in aux_list if region in x['serviceRegions']]

            for a in aux_list:
                instance_data_gcp.append(a)

        if 'Core' in instance_data_gcp[0]['description']:
            int_price_per_vcpu = int(instance_data_gcp[0]['pricingInfo'][0]
                                     ['pricingExpression']['tieredRates']
                                     [0]['unitPrice']['units'])
            cents_per_vcpu = int(instance_data_gcp[0]['pricingInfo'][0]
                                 ['pricingExpression']['tieredRates'][0]
                                 ['unitPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            int_price_per_ram = int(instance_data_gcp[1]['pricingInfo'][0]
                                    ['pricingExpression']['tieredRates'][0]
                                    ['unitPrice']['units'])
            cents_per_ram = int(instance_data_gcp[1]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates'][0]
                                ['unitPrice']['nanos']) / 1000000000
            price_per_ram = int_price_per_ram + cents_per_ram
        else:
            int_price_per_vcpu = int(instance_data_gcp[1]['pricingInfo'][0]
                                     ['pricingExpression']['tieredRates']
                                     [0]['unitPrice']['units'])
            cents_per_vcpu = int(instance_data_gcp[1]['pricingInfo'][0]
                                 ['pricingExpression']['tieredRates'][0]
                                 ['unitPrice']['nanos']) / 1000000000
            price_per_vcpu = int_price_per_vcpu + cents_per_vcpu
            int_price_per_ram = int(instance_data_gcp[0]['pricingInfo'][0]
                                    ['pricingExpression']['tieredRates'][0]
                                    ['unitPrice']['units'])
            cents_per_ram = int(instance_data_gcp[0]['pricingInfo'][0]
                                ['pricingExpression']['tieredRates'][0]
                                ['unitPrice']['nanos']) / 1000000000
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
        request = self.compute_engine.regions().list(project=self.gcp_conf.project, filter=f'(description = {region})')

        zones = []

        if request is not None:
            response = request.execute()

            for az in response['items'][0]['zones']:
                zones.append(az.split('/')[-1])

        return zones

    # def get_cpu_credits(self, instance_id):
    #
    #     end_time = datetime.utcnow()
    #
    #     start_time = end_time - timedelta(minutes=5)
    #
    #     # print(start_time)
    #     # print(end_time)
    #
    #     response = self.cloud_watch.get_metric_statistics(
    #         Namespace='AWS/EC2',
    #         MetricName='CPUCreditBalance',
    #         Dimensions=[
    #             {
    #                 'Name': 'InstanceId',
    #                 'Value': instance_id
    #             },
    #         ],
    #         Period=60,
    #         Statistics=['Average'],
    #         StartTime=start_time,
    #         EndTime=end_time
    #     )
    #
    #     cpu_credits = 0
    #
    #     if len(response['Datapoints']) > 0:
    #         cpu_credits = math.ceil(response['Datapoints'][-1]['Average'])
    #
    #     return cpu_credits
