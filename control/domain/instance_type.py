from control.config.ec2_config import EC2Config


class InstanceType:

    def __init__(self, provider, instance_type, image_id, prices, restrictions,
                 vcpu, gpu, count_gpu, memory, ebs_device_name=''):
        self.provider = provider
        self.type = instance_type
        self.vcpu = vcpu
        self.memory = memory
        # self.gflops = gflops
        self.price_ondemand = prices['on-demand']
        self.price_preemptible = prices['preemptible']
        self.restrictions = restrictions
        self.image_id = image_id
        self.ebs_device_name = ebs_device_name
        self.gpu = gpu
        self.count_gpu = count_gpu

        self.id = None

        # TODO change this to reflect GCP as well
        config = EC2Config()

        self.boot_overhead_seconds = config.boot_overhead
        self.interruption_overhead_seconds = config.interruption_overhead

        self.region = None
        self.zone = None

    def setup_ondemand_price(self, price, region):
        self.price_ondemand = price
        self.region = region

    def setup_preemptible_price(self, price, region, zone):
        self.price_preemptible = price
        self.region = region
        self.zone = zone

    @classmethod
    def from_dict(cls, adict):
        return [
            cls(
                provider=adict['instances'][key]['provider'],
                instance_type=key,
                image_id=adict['instances'][key]['image_id'],
                ebs_device_name=adict['instances'][key]['ebs_device_name'],
                prices=adict['instances'][key]['prices'],
                vcpu=adict['instances'][key]['vcpu'],
                restrictions=adict['instances'][key]['restrictions'],
                gpu=adict['instances'][key]['gpu'],
                memory=adict['instances'][key]['memory'],
                count_gpu=adict['instances'][key]['gpu_count']
            )
            for key in adict['instances']
        ]

    # @property
    # def rank(self):
    #     return self.gflops / self.price_preemptible

    @property
    def market_ondemand(self):
        return self.restrictions['markets']['on-demand'].lower() in ['yes']

    @property
    def market_preemptible(self):
        return self.restrictions['markets']['preemptible'].lower() in ['yes']

    @property
    def limits_ondemand(self):
        return self.restrictions['limits']['on-demand']

    @property
    def limits_preemptible(self):
        return self.restrictions['limits']['preemptible']

    @property
    def have_gpu(self):
        return self.count_gpu > 0

    def __str__(self):
        return "'{}' on-demand price: '{}' preemptible price: '{}' " \
               "region: '{}' zone: '{}' provider: {} GPU? {}".format(self.type,
                                                                     self.price_ondemand,
                                                                     self.price_preemptible,
                                                                     self.region,
                                                                     self.zone,
                                                                     self.provider,
                                                                     self.gpu
                                                                     )
