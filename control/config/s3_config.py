from control.config.config import Config


class S3Config(Config):
    _key = 's3'

    @property
    def bucket_name(self):
        return self.get_property(self._key, 'bucket_name')

    @property
    def vm_uid(self):
        return self.get_property(self._key, 'vm_uid')

    @property
    def vm_gid(self):
        return self.get_property(self._key, 'vm_gid')

    @property
    def path(self):
        return self.get_property(self._key, 'path')
