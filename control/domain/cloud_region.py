class CloudRegion:

    def __init__(self, region_id, provider, region, zones, client_image_id, server_image_id, key_file):
        self.id = region_id

        self.provider = provider
        self.region = region
        self.zones = zones
        self.client_image_id = client_image_id
        self.server_image_id = server_image_id
        self.key_file = key_file

    def setup_zones(self, zones):
        self.zones = zones

    @classmethod
    def from_dict(cls, adict):
        return [
            cls(
                region_id=key,
                provider=adict['locations'][key]['provider'],
                region=adict['locations'][key]['region'],
                client_image_id=adict['locations'][key]['client_image_id'],
                server_image_id=adict['locations'][key]['server_image_id'],
                zones=adict['locations'][key]['zones'],
                key_file=adict['locations'][key]['key_file']
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
