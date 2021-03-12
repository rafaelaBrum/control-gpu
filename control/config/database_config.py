from control.config.config import Config


class DataBaseConfig(Config):
    _key = 'database'

    @property
    def user(self):
        return self.get_property(self._key, 'user')

    @property
    def password(self):
        return self.get_property(self._key, 'password')

    @property
    def host(self):
        return self.get_property(self._key, 'host')

    @property
    def database_name(self):
        return self.get_property(self._key, 'database_name')

    @property
    def dump_dir(self):
        return self.get_property(self._key, 'dump_dir')

    @property
    def with_dump(self):
        return self.get_boolean(self._key, 'with_dump')
