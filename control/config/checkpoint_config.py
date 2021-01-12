from control.config.config import Config


class CheckPointConfig(Config):
    _key = 'checkpoint'

    @property
    def with_checkpoint(self):
        return self.get_boolean(self._key, 'with_checkpoint')





