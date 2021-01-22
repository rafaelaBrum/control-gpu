from control.config.config import Config


class EC2Config(Config):
    _key = 'ec2'

    @property
    def security_group(self):
        return self.get_property(self._key, 'security_group')

    @property
    def security_vpc_group(self):
        return self.get_property(self._key, 'security_vpc_group')

    @property
    def key_name(self):
        return self.get_property(self._key, 'key_name')

    @property
    def image_id(self):
        return self.get_property(self._key, 'image_id')

    @property
    def tag_key(self):
        return self.get_property(self._key, 'tag_key')

    @property
    def tag_value(self):
        return self.get_property(self._key, 'tag_value')

    @property
    def zone(self):
        return self.get_property(self._key, 'zone')

    @property
    def region(self):
        return self.get_property(self._key, 'region')

    @property
    def home_path(self):
        return self.get_property(self._key, 'home_path')

    @property
    def input_path(self):
        return self.get_property(self._key, 'input_path')

    @property
    def boot_overhead(self):
        return float(self.get_property(self._key, 'boot_overhead'))

    @property
    def interruption_overhead(self):
        return float(self.get_property(self._key, 'interruption_overhead'))

    @property
    def vcpu_limits(self):
        return int(self.get_property(self._key, 'vcpu_limits'))

    @property
    def vm_user(self):
        return self.get_property(self._key, 'vm_user')
