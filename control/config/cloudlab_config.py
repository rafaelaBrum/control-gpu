from control.config.config import Config


class CloudLabConfig(Config):
    _key = 'cloudlab'

    @property
    def project_name(self):
        return self.get_property(self._key, 'project_name')

    @property
    def server_experiment_name(self):
        return self.get_property(self._key, 'server_experiment_name')

    @property
    def client_experiment_name(self):
        return self.get_property(self._key, 'client_experiment_name')
