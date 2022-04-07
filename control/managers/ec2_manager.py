from control.managers.cloud_manager import CloudManager

from control.config.ec2_config import EC2Config
from control.config.s3_config import S3Config

import boto3

from datetime import datetime
from dateutil.tz import tzutc
# from datetime import timedelta

import logging
import json

# import math

from pkg_resources import resource_filename

from ratelimit import limits, sleep_and_retry

from botocore.config import Config


def get_ec2_client(region):
    if region != '':
        client = boto3.client('ec2', region_name=region)
    else:
        client = boto3.client('ec2')
    return client


def get_ec2_resource(region):
    if region != '':
        client = boto3.resource('ec2', region_name=region)
    else:
        client = boto3.resource('ec2')
    return client


def get_cloudwatch_client(region):
    if region != '':
        client = boto3.client('cloudwatch', region_name=region)
    else:
        client = boto3.client('cloudwatch')
    return client


def _new_filter(name, values):
    return {
        'Name': name,
        'Values': [v for v in values]
    }


class EC2Manager(CloudManager):
    ec2_conf = EC2Config()
    s3_conf = S3Config()
    vm_config = ec2_conf
    bucket_config = s3_conf

    session = boto3.Session()
    credentials = session.get_credentials()

    def __init__(self):

        self.instances_history = {}

    def _update_history(self, instances, status):
        for i in instances:

            if status == 'start':
                try:
                    i.load()
                except:
                    pass

                self.instances_history = {
                    i.id: {
                        'StartTime': i.launch_time,
                        'EndTime': None,
                        'Instance': i,
                        'Zone': i.placement['AvailabilityZone']
                    }
                }

            if status == 'terminate':
                if i.id in self.instances_history:
                    self.instances_history[i.id]['EndTime'] = \
                        datetime.now(tz=tzutc())

    def _create_instance(self, info, burstable, region):

        try:
            resource = get_ec2_resource(region)

            if burstable:

                instances = resource.create_instances(
                    ImageId=info['ImageId'],
                    InstanceType=info['InstanceType'],
                    KeyName=info['KeyName'],
                    MaxCount=info['MaxCount'],
                    MinCount=info['MinCount'],
                    SecurityGroups=info['SecurityGroups'],
                    InstanceMarketOptions=info['InstanceMarketOptions'],
                    TagSpecifications=[
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {
                                    'Key': self.ec2_conf.tag_key,
                                    'Value': self.ec2_conf.tag_value
                                }
                            ]
                        }
                    ],
                    Placement=info['Placement'],
                    CreditSpecification={
                        'CpuCredits': 'standard'
                    }
                )

            else:
                instances = resource.create_instances(
                    ImageId=info['ImageId'],
                    InstanceType=info['InstanceType'],
                    KeyName=info['KeyName'],
                    MaxCount=info['MaxCount'],
                    MinCount=info['MinCount'],
                    SecurityGroups=info['SecurityGroups'],
                    InstanceMarketOptions=info['InstanceMarketOptions'],
                    TagSpecifications=[
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {
                                    'Key': self.ec2_conf.tag_key,
                                    'Value': self.ec2_conf.tag_value
                                }
                            ]
                        }
                    ],
                    Placement=info['Placement']
                )

            self._update_history(instances, 'start')

            for i in instances:
                i.wait_until_running()

            return instances

        except Exception as e:
            logging.error("<EC2Manager>: Error to create instance")
            logging.error(e)
            return None

    def create_volume(self, size, volume_name='', zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        try:
            client = get_ec2_client(region)

            ebs_vol = client.create_volume(
                Size=size,
                AvailabilityZone=self.ec2_conf.zone,
                TagSpecifications=[
                    {
                        'ResourceType': 'volume',
                        'Tags': [
                            {
                                'Key': self.ec2_conf.tag_key,
                                'Value': self.ec2_conf.tag_value
                            },
                        ]
                    },
                ],
            )

            if ebs_vol['ResponseMetadata']['HTTPStatusCode'] == 200:

                return ebs_vol['VolumeId']
            else:
                return None

        except Exception as e:

            logging.error("<EC2Manager>: Error to create Volume")
            logging.error(e)
            return None

    def wait_volume(self, volume_id, zone):

        region = zone
        if region != '':
            region = zone[:-1]

        client = get_ec2_client(region)

        waiter = client.get_waiter('volume_available')
        waiter.wait(
            VolumeIds=[
                volume_id
            ]
        )

    def attach_volume(self, instance_id, volume_id, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        try:
            client = get_ec2_client(region)

            client.attach_volume(
                VolumeId=volume_id,
                InstanceId=instance_id,
                Device="/dev/xvdf"
            )
            return True
        except Exception as e:
            logging.error("<EC2Manager>: Error to attach volume {} to instance {}".format(volume_id,
                                                                                          instance_id))
            logging.error(e)
            return False

    def create_on_demand_instance(self, instance_type, image_id, zone='', burstable=False, key_name=''):

        region = ''
        if zone != '':
            parameters = {

                'ImageId': image_id,
                'InstanceType': instance_type,
                'KeyName': self.ec2_conf.key_name,
                'MaxCount': 1,
                'MinCount': 1,
                'SecurityGroups': [
                    self.ec2_conf.security_group,
                    self.ec2_conf.security_vpc_group
                ],
                'InstanceMarketOptions': {},
                'Placement': {'AvailabilityZone': zone},
            }
            region = zone[:-1]
        else:
            parameters = {

                'ImageId': image_id,
                'InstanceType': instance_type,
                'KeyName': self.ec2_conf.key_name,
                'MaxCount': 1,
                'MinCount': 1,
                'SecurityGroups': [
                    self.ec2_conf.security_group,
                    self.ec2_conf.security_vpc_group
                ],
                'InstanceMarketOptions': {},
                'Placement': {'AvailabilityZone': self.ec2_conf.zone},
            }

        if key_name != '':
            parameters['KeyName'] = key_name

        instances = self._create_instance(parameters, burstable, region=region)

        if instances is not None:
            created_instances = [i for i in instances]
            instance = created_instances[0]
            return instance.id
        else:
            return None

    def delete_volume(self, volume_id, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        try:
            client = get_ec2_client(region)

            client.delete_volume(VolumeId=volume_id)
            status = True
        except Exception as e:
            logging.error("<EC2Manager>: Error to delete Volume {} ".format(volume_id))
            logging.error(e)
            status = False

        return status

    def create_preemptible_instance(self, instance_type, image_id, max_price, zone='', burstable=False):

        # user_data = '''#!/bin/bash
        # /usr/bin/enable-ec2-spot-hibernation
        # echo "user-data" > $HOME/control-applications/test_user_data.txt
        # '''

        zone = self.ec2_conf.zone
        region = zone[:-1]
        interruption_behaviour = 'stop'

        parameters = {
            'ImageId': image_id,
            'InstanceType': instance_type,
            'KeyName': self.ec2_conf.key_name,
            'MaxCount': 1,
            'MinCount': 1,
            'SecurityGroups': [
                self.ec2_conf.security_group,
                self.ec2_conf.security_vpc_group
            ],
            'InstanceMarketOptions':
                {
                    'MarketType': 'spot',
                    'SpotOptions': {
                        'MaxPrice': str(max_price),
                        'SpotInstanceType': 'persistent',
                        'InstanceInterruptionBehavior': interruption_behaviour
                    }
                },
            'Placement': {}
            # 'UserData': user_data
        }

        if zone:
            parameters['Placement'] = {'AvailabilityZone': zone}

        instances = self._create_instance(parameters, burstable, region)

        if instances is not None:
            created_instances = [i for i in instances]
            instance = created_instances[0]
            return instance.id
        else:
            return None

    def _terminate_instance(self, instance, region):

        # if instance is spot, we have to remove its request
        if instance.instance_lifecycle == 'spot':
            client = get_ec2_client(region)

            client.cancel_spot_instance_requests(
                SpotInstanceRequestIds=[
                    instance.spot_instance_request_id
                ]
            )

        self._update_history([instance], status='terminate')

        instance.terminate()

    def terminate_instance(self, instance_id, wait=True, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        try:

            instance = self.__get_instance(instance_id, region=region)

            self._terminate_instance(instance, region=region)

            if wait:
                instance.wait_until_terminated()

            status = True

        except Exception as e:
            logging.error("<EC2Manager>: Error to terminate instance {}".format(instance_id))
            logging.error(e)

            status = False

        return status

    @sleep_and_retry
    @limits(calls=10, period=1)
    def __get_instance(self, instance_id, region):

        # print("instance_id", instance_id)
        # print("region", region)

        try:
            resource = get_ec2_resource(region)

            instance = resource.Instance(instance_id)

            instance.reload()
        except Exception as e:
            logging.info(e)
            return None

        return instance

    def __get_instances(self, search_filter=None, region=''):

        if search_filter is None:
            resource = get_ec2_resource(region)
            return resource.instances.filter()

        _filters = []

        if 'status' in search_filter:
            _filters.append(
                _new_filter(
                    name="instance-state-name",
                    values=search_filter['status']
                )
            )

        if 'tag' in search_filter:
            _filters.append(
                _new_filter(
                    name='tag:{}'.format(search_filter['tag']['name']),
                    values=search_filter['tag']['values']
                )
            )

        resource = get_ec2_resource(region)

        return [i for i in resource.instances.filter(Filters=_filters)]

    def get_instance_status(self, instance_id, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        if instance_id is None:
            return None

        instance = self.__get_instance(instance_id, region)

        if instance is None:
            return None
        else:
            return instance.state["Name"].lower()

    def list_instances_id(self, search_filter=None, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        instances = self.__get_instances(search_filter, region=region)

        return [i.id for i in instances]

    def get_public_instance_ip(self, instance_id, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        instance = self.__get_instance(instance_id, region)

        if instance is None:
            return None
        else:
            return instance.public_ip_address

    def get_private_instance_ip(self, instance_id, zone=''):

        region = zone
        if region != '':
            region = zone[:-1]

        instance = self.__get_instance(instance_id, region)

        if instance is None:
            return None
        else:
            return instance.private_ip_address

    @staticmethod
    def get_preemptible_price(instance_type, zone=None, region=''):

        _filters = [
            {
                'Name': 'product-description',
                'Values': ['Linux/UNIX']
            }
        ]

        if zone is not None:
            _filters.append(
                {
                    'Name': 'availability-zone',
                    'Values': [zone]
                }
            )

        client = get_ec2_client(region=region)

        history = client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            Filters=_filters,
            StartTime=datetime.now()
        )

        return [(h['AvailabilityZone'], float(h['SpotPrice'])) for h in history['SpotPriceHistory']]

    # Get current AWS price for an on-demand instance
    @staticmethod
    def get_ondemand_price(instance_type, region):
        # Search product search_filter
        FLT = '[{{"Field": "tenancy", "Value": "shared", "Type": "TERM_MATCH"}},' \
              '{{"Field": "operatingSystem", "Value": "Linux", "Type": "TERM_MATCH"}},' \
              '{{"Field": "preInstalledSw", "Value": "NA", "Type": "TERM_MATCH"}},' \
              '{{"Field": "instanceType", "Value": "{t}", "Type": "TERM_MATCH"}},' \
              '{{"Field": "location", "Value": "{r}", "Type": "TERM_MATCH"}},' \
              '{{"Field": "capacitystatus", "Value": "Used", "Type": "TERM_MATCH"}}]'

        # translate region code to region name
        endpoint_file = resource_filename('botocore', 'data/endpoints.json')

        with open(endpoint_file, 'r') as f:
            data = json.load(f)
        region = data['partitions'][0]['regions'][region]['description']

        f = FLT.format(r=region, t=instance_type)
        # Get price info
        pricing = boto3.client('pricing')
        data = pricing.get_products(ServiceCode='AmazonEC2', Filters=json.loads(f))
        od = json.loads(data['PriceList'][0])['terms']['OnDemand']
        id1 = list(od)[0]
        id2 = list(od[id1]['priceDimensions'])[0]

        return float(od[id1]['priceDimensions'][id2]['pricePerUnit']['USD'])

    # Get availability zones of a AWS region
    @staticmethod
    def get_availability_zones(region):
        # Get zones info
        my_config = Config(region_name=region)
        ec2 = boto3.client('ec2', config=my_config)
        data = ec2.describe_availability_zones()

        zones = []

        for az in data['AvailabilityZones']:
            if az['State'] == 'available':
                zones.append(az['ZoneName'])

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
