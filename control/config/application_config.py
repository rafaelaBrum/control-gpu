from control.config.config import Config


class ApplicationConfig(Config):
    _key = 'application'

    @property
    def app_local_path(self):
        return self.get_property(self._key, 'app_local_path')

    @property
    def daemon_path(self):
        return self.get_property(self._key, 'daemon_path')

    @property
    def daemon_file(self):
        return self.get_property(self._key, 'daemon_file')

    @property
    def flower_path(self):
        return self.get_property(self._key, 'flower_path')

    @property
    def server_flower_file(self):
        return self.get_property(self._key, 'server_flower_file')

    @property
    def client_flower_file(self):
        return self.get_property(self._key, 'client_flower_file')

    @property
    def centralized_path(self):
        return self.get_property(self._key, 'centralized_path')

    @property
    def centralized_file(self):
        return self.get_property(self._key, 'centralized_file')

    @property
    def centralized_app_path(self):
        return self.get_property(self._key, 'centralized_app_path')

    @property
    def centralized_app_file(self):
        return self.get_property(self._key, 'centralized_app_file')

    @property
    def data_path(self):
        return self.get_property(self._key, 'data_path')

    @property
    def dataset(self):
        return self.get_property(self._key, 'dataset')