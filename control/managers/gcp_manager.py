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


# TODO: get API key from GClod Python API
file = open(Path(Path.home(), 'gcloud_api_key'), 'r')
api_key = file.read()


class GCPManager(CloudManager):

    def __init__(self):

        self.gcp_conf = GCPConfig()
        self.storage_config = StorageConfig()
        self.vm_config = self.gcp_conf

        credentials = GoogleCredentials.get_application_default()
        self.compute_engine = googleapiclient.discovery.build('compute', 'v1',
                                                              credentials=credentials, cache_discovery=False)
        # self.cloud_watch = boto3.client('cloudwatch')
        # self.session = boto3.Session()
        # self.credentials = GoogleCredentials.get_application_default()

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

    def _wait_for_operation(self, operation):
        # print('Waiting for operation to finish...')

        while True:
            result = self.compute_engine.zoneOperations().get(
                project=self.gcp_conf.project,
                zone=self.gcp_conf.zone,
                operation=operation).execute()

            if result['status'] == 'DONE':
                # print("done.")
                if 'error' in result:
                    raise Exception(result['error'])
                return result

            time.sleep(1)

    def _create_instance(self, info):

        try:
            operation = self.compute_engine.instances().insert(
                project=self.gcp_conf.project,
                zone=self.gcp_conf.zone,
                body=info).execute()

            self._wait_for_operation(operation['name'])

            instances = self.__get_instances(filter=f'(name = {info["name"]})')

            self._update_history(instances, 'start')

            return instances

        except Exception as e:
            logging.error("<GCPManager>: Error to create instance")
            logging.error(e)
            return None

    def create_volume(self, size, volume_name=''):
        try:
            disk_body = {
                'name': volume_name,
                "sizeGb": size,
                'type': f'projects/{self.gcp_conf.project}/zones/{self.gcp_conf.zone}/diskTypes/pd-balanced'
            }

            operation = self.compute_engine.disks().insert(project=self.gcp_conf.project, zone=self.gcp_conf.zone,
                                                           body=disk_body).execute()

            self._wait_for_operation(operation['name'])

            disk = self.__get_disk(volume_name)

            return disk['id'] if disk else None

        except Exception as e:

            logging.error("<GCPManager>: Error to create Volume")
            logging.error(e)
            return None

    def wait_volume(self, volume_name=''):
        disk = self.__get_disk(volume_name)

        ready = False

        while disk is not None and not ready:
            if 'lastAttachTimestamp' not in disk:
                ready = True
            elif 'lastDetachTimestamp' in disk:
                last_attach_time = iso8601.parse_date(disk['lastAttachTimestamp'])
                last_detach_time = iso8601.parse_date(disk['lastDetachTimestamp'])
                ready = last_detach_time > last_attach_time
            if not ready:
                disk = self.__get_disk(volume_name)

    def attach_volume(self, instance_id, volume_id, volume_name=''):

        try:
            instance = self.__get_instance(instance_id)

            disk = self.compute_engine.disks().get(project=self.gcp_conf.project, zone=self.gcp_conf.zone,
                                                   disk=volume_name).execute()

            if disk is not None:
                attached_disk_body = {
                    'source': disk['selfLink']
                }

                operation = self.compute_engine.instances().attachDisk(project=self.gcp_conf.project,
                                                                       zone=self.gcp_conf.zone,
                                                                       instance=instance['name'],
                                                                       body=attached_disk_body).execute()

                self._wait_for_operation(operation['name'])

                return True
            else:
                return False
        except Exception as e:
            logging.error("<GCPManager>: Error to attach volume {} ({}} to instance {}".format(volume_id,
                                                                                               volume_name,
                                                                                               instance_id))
            logging.error(e)
            return False

    def create_on_demand_instance(self, instance_type, image_id, vm_name=''):

        machine_type = f'zones/{self.gcp_conf.zone}/machineTypes/n2-standard-2'

        image_response = self.compute_engine.images().get(project=self.gcp_conf.project,
                                                          image=f'{image_id}').execute()
        source_disk_image = image_response['selfLink']

        config = {
            'name': vm_name,
            'machineType': machine_type,

            # Not working. Still in Beta on GCP API!
            # # 'soourceMachineImage': f'projects/{self.gcp_conf.project}/machineImages/{image_id}',
            # 'soourceMachineImage': source_machine_image,

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
        }

        instances = self._create_instance(config)

        if instances is not None:
            created_instances = [i for i in instances]
            instance = created_instances[0]
            return instance['id']
        else:
            return None

    def delete_volume(self, volume_id, volume_name=''):
        try:
            self.compute_engine.disks().delete(project=self.gcp_conf.project, zone=self.gcp_conf.zone,
                                               disk=volume_name).execute()
            status = True
        except Exception as e:
            logging.error("<GCPManager>: Error to delete Volume {} ({}) ".format(volume_id, volume_name))
            logging.error(e)
            status = False

        return status

    def create_preemptible_instance(self, instance_type, image_id, max_price, burstable=False):
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

    def _terminate_instance(self, instance):
        # if instance is spot, we have to remove its request
        # if instance.instance_lifecycle == 'spot':
        #     self.client.cancel_spot_instance_requests(
        #         SpotInstanceRequestIds=[
        #             instance.spot_instance_request_id
        #         ]
        #     )

        self._update_history([instance], status='terminate')

        operation = self.compute_engine.instances().delete(project=self.gcp_conf.project,
                                                           zone=self.gcp_conf.zone,
                                                           instance=instance['name']).execute()

        return operation

    def terminate_instance(self, instance_id, wait=True):
        try:

            instance = self.__get_instance(instance_id)

            operation = self._terminate_instance(instance)

            if wait:
                self._wait_for_operation(operation['name'])

            status = True

        except Exception as e:
            logging.error("<GCPManager>: Error to terminate instance {}".format(instance_id))
            logging.error(e)

            status = False

        return status

    @sleep_and_retry
    @limits(calls=10, period=1)
    def __get_instance(self, instance_id):
        try:

            instance = self.__get_instances(filter=f'(id = {instance_id})')[0]

        except Exception as e:
            logging.info(e)
            return None

        return instance

    def __get_instances(self, filter=None):

        if filter is None:
            result = self.compute_engine.instances().list(project=self.gcp_conf.project,
                                                          zone=self.gcp_conf.zone).execute()
        else:
            result = self.compute_engine.instances().list(project=self.gcp_conf.project,
                                                          zone=self.gcp_conf.zone,
                                                          filter=filter).execute()
        return result['items'] if 'items' in result else None

    def get_instance_status(self, instance_id):
        if instance_id is None:
            return None

        instance = self.__get_instances(filter=f'(id = {instance_id})')[0]

        if instance is None:
            return None
        else:
            # print("instance status", instance['status'])

            return instance['status']

    def __get_disk(self, disk_name):
        try:
            return self.compute_engine.disks().get(project=self.gcp_conf.project,
                                                   zone=self.gcp_conf.zone, disk=disk_name).execute()
        except Exception as e:
            logging.error("<GCPManager>: Error to find instance")
            logging.error(e)
            return None

    def list_instances_id(self, search_filter=None):
        instances = self.__get_instances(search_filter)

        return [i['id'] for i in instances] if instances else []

    def get_public_instance_ip(self, instance_id):
        instance = self.__get_instances(filter=f'(id = {instance_id})')[0]
        if instance is None:
            return None
        else:
            return instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']

    def get_private_instance_ip(self, instance_id):
        instance = self.__get_instances(filter=f'(id = {instance_id})')[0]
        if instance is None:
            return None
        else:
            return instance['networkInterfaces'][0]['networkIP']

    @staticmethod
    def get_preemptible_price(instance_type, zone=None):
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
                        if f'{instance_type.split("-")[0].upper()} Instance' in x['description']
                        and 'Preemptible' in x['description']]

            if 'us' in zone:
                aux_list = [x for x in aux_list if 'Americas' in x['description']]

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
                        if f'{instance_type.split("-")[0].upper()} Instance' in x['description']
                        and 'Preemptible' not in x['description']]

            if 'us' in region:
                aux_list = [x for x in aux_list if 'Americas' in x['description']]

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
