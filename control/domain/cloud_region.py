class CloudRegion:

    def __init__(self, id, provider, region, zones):
        self.id = id

        self.provider = provider
        self.region = region
        self.zones = zones

    def setup_zones(self, zones):
        self.zones = zones

    @classmethod
    def from_dict(cls, adict):
        return [
            cls(
                id=key,
                provider=adict['locations'][key]['provider'],
                region=adict['locations'][key]['region'],
                zones=adict['locations'][key]['zones']
            )
            for key in adict['locations']
        ]

    @property
    def count_zones(self):
        return len(self.zones)

    def __str__(self):
        return "'{}' provider: '{}' region: '{}' zones: '{}' ".format(self.id,
                                                                      self.provider,
                                                                      self.region,
                                                                      self.zones)
