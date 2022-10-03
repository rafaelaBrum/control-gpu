from control.config.config import Config


class CloudLabConfig(Config):
    _key = 'cloudlab'

    @property
    def project_name(self):
        return self.get_property(self._key, 'project_name')
