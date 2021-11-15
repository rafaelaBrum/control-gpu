from control.config.config import Config


class GCPConfig(Config):
    _key = 'gcp'

    @property
    def project(self):
        return self.get_property(self._key, 'project')

    @property
    def zone(self):
        return self.get_property(self._key, 'zone')

    @property
    def region(self):
        return self.get_property(self._key, 'region')
