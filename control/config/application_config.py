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
    def daemon_aws_file(self):
        return self.get_property(self._key, 'daemon_aws_file')

    @property
    def daemon_gcp_file(self):
        return self.get_property(self._key, 'daemon_gcp_file')

    @property
    def fl_framework(self):
        return self.get_property(self._key, 'fl_framework')

    @property
    def fl_port(self):
        return self.get_property(self._key, 'fl_port')

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
