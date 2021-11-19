from pprint import pprint

from control.managers.cloud_manager import CloudManager

from control.config.gcp_config import GCPConfig

import googleapiclient.discovery

from datetime import datetime
from dateutil.tz import tzutc
from datetime import timedelta

import logging
import json

import math
import time

from pkg_resources import resource_filename

from ratelimit import limits, sleep_and_retry


class GCPManager(CloudManager):

    def __init__(self):

        self.gcp_conf = GCPConfig()

        self.compute_engine = googleapiclient.discovery.build('compute', 'v1')
        # self.resource = boto3.resource('ec2')
        # self.cloud_watch = boto3.client('cloudwatch')
        # self.session = boto3.Session()
        # self.credentials = self.session.get_credentials()

        self.instances_history = {}

    # def _new_filter(self, name, values):
    #     return {
    #         'Name': name,
    #         'Values': [v for v in values]
    #     }

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

    # def create_volume(self, size, zone):
    #
    #     try:
    #         ebs_vol = self.client.create_volume(
    #             Size=size,
    #             AvailabilityZone=zone,
    #             TagSpecifications=[
    #                 {
    #                     'ResourceType': 'volume',
    #                     'Tags': [
    #                         {
    #                             'Key': self.ec2_conf.tag_key,
    #                             'Value': self.ec2_conf.tag_value
    #                         },
    #                     ]
    #                 },
    #             ],
    #         )
    #
    #         if ebs_vol['ResponseMetadata']['HTTPStatusCode'] == 200:
    #
    #             return ebs_vol['VolumeId']
    #         else:
    #             return None
    #
    #     except Exception as e:
    #
    #         logging.error("<GCPManager>: Error to create Volume")
    #         logging.error(e)
    #         return None

    # def wait_volume(self, volume_id):
    #     waiter = self.client.get_waiter('volume_available')
    #     waiter.wait(
    #         VolumeIds=[
    #             volume_id
    #         ]
    #     )

    # def attach_volume(self, instance_id, volume_id, device):
    #
    #     try:
    #         self.client.attach_volume(
    #             VolumeId=volume_id,
    #             InstanceId=instance_id,
    #             Device="/dev/xvdf"
    #         )
    #         return True
    #     except Exception as e:
    #         logging.error("<GCPManager>: Error to attach volume {} to instance {}".format(volume_id,
    #                                                                                       instance_id))
    #         logging.error(e)
    #         return False

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

    # def delete_volume(self, volume_id):
    #     try:
    #         self.client.delete_volume(VolumeId=volume_id)
    #         status = True
    #     except Exception as e:
    #         logging.error("<GCPManager>: Error to delete Volume {} ".format(volume_id))
    #         logging.error(e)
    #         status = False
    #
    #     return status

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

    def _terminate_instance(self, instance, instance_name):
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

            # print("instance")
            # print(instance['name'])

            # return True

            operation = self._terminate_instance(instance, instance['id'])

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

        # if search_filter is None:
        #     return self.resource.instances.filter()
        #
        # _filters = []
        #
        # if 'status' in search_filter:
        #     _filters.append(
        #         self._new_filter(
        #             name="instance-state-name",
        #             values=search_filter['status']
        #         )
        #     )
        #
        # if 'tag' in search_filter:
        #     _filters.append(
        #         self._new_filter(
        #             name='tag:{}'.format(search_filter['tag']['name']),
        #             values=search_filter['tag']['values']
        #         )
        #     )
        #
        # return [i for i in self.resource.instances.filter(Filters=_filters)]

    def get_instance_status(self, instance_id):
        if instance_id is None:
            return None

        instance = self.__get_instances(filter=f'(id = {instance_id})')[0]

        if instance is None:
            return None
        else:
            # print("instance status", instance['status'])

            return instance['status']

        # if instance_id is None:
        #     return None
        #
        # instance = self.__get_instance(instance_id)
        #
        # if instance is None:
        #     return None
        # else:
        #     return instance.state["Name"].lower()

    def list_instances_id(self, search_filter=None):
        pass
        # instances = self.__get_instances(search_filter)
        #
        # return [i.id for i in instances]

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
        pass

        # _filters = [
        #     {
        #         'Name': 'product-description',
        #         'Values': ['Linux/UNIX']
        #     }
        # ]
        #
        # if zone is not None:
        #     _filters.append(
        #         {
        #             'Name': 'availability-zone',
        #             'Values': [zone]
        #         }
        #     )
        #
        # client = boto3.client('ec2')
        #
        # history = client.describe_spot_price_history(
        #     InstanceTypes=[instance_type],
        #     Filters=_filters,
        #     StartTime=datetime.now()
        # )
        #
        # return [(h['AvailabilityZone'], float(h['SpotPrice'])) for h in history['SpotPriceHistory']]

    # Get current AWS price for an on-demand instance
    @staticmethod
    def get_ondemand_price(instance_type, region):
        pass
        # Search product search_filter
        # FLT = '[{{"Field": "tenancy", "Value": "shared", "Type": "TERM_MATCH"}},' \
        #       '{{"Field": "operatingSystem", "Value": "Linux", "Type": "TERM_MATCH"}},' \
        #       '{{"Field": "preInstalledSw", "Value": "NA", "Type": "TERM_MATCH"}},' \
        #       '{{"Field": "instanceType", "Value": "{t}", "Type": "TERM_MATCH"}},' \
        #       '{{"Field": "location", "Value": "{r}", "Type": "TERM_MATCH"}},' \
        #       '{{"Field": "capacitystatus", "Value": "Used", "Type": "TERM_MATCH"}}]'
        #
        # # translate region code to region name
        # endpoint_file = resource_filename('botocore', 'data/endpoints.json')
        #
        # with open(endpoint_file, 'r') as f:
        #     data = json.load(f)
        # region = data['partitions'][0]['regions'][region]['description']
        #
        # f = FLT.format(r=region, t=instance_type)
        # # Get price info
        # pricing = boto3.client('pricing')
        # data = pricing.get_products(ServiceCode='AmazonEC2', Filters=json.loads(f))
        # od = json.loads(data['PriceList'][0])['terms']['OnDemand']
        # id1 = list(od)[0]
        # id2 = list(od[id1]['priceDimensions'])[0]
        #
        # return float(od[id1]['priceDimensions'][id2]['pricePerUnit']['USD'])

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
