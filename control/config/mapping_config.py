from control.config.config import Config


class MappingConfig(Config):
    _key = 'mapping'

    @property
    def budget(self):
        return self.get_property(self._key, 'budget')

    @property
    def deadline(self):
        return self.get_property(self._key, 'deadline')

    @property
    def alpha(self):
        return self.get_property(self._key, 'alpha')

    @property
    def scheduler_name(self):
        return self.get_property(self._key, 'scheduler_name')