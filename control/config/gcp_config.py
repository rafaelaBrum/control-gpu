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

    @property
    def home_path(self):
        return self.get_property(self._key, 'home_path')

    @property
    def input_path(self):
        return self.get_property(self._key, 'input_path')

    @property
    def key_path(self):
        return self.get_property(self._key, 'key_path')

    @property
    def key_file(self):
        return self.get_property(self._key, 'key_file')

    @property
    def vm_user(self):
        return self.get_property(self._key, 'vm_user')

    @property
    def credentials_file(self):
        return self.get_property(self._key, 'credentials_file')

    @property
    def aws_settings(self):
        return self.get_property(self._key, 'aws_settings')

    @property
    def gid(self):
        return self.get_property(self._key, 'gid')

    @property
    def uid(self):
        return self.get_property(self._key, 'uid')
