from control.config.config import Config


class StorageConfig(Config):
    _key = 'storage'

    @property
    def bucket_name(self):
        return self.get_property(self._key, 'bucket_name')
