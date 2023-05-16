from control.config.config import Config


class CheckPointConfig(Config):
    _key = 'checkpoint'

    @property
    def with_checkpoint(self):
        return self.get_boolean(self._key, 'with_checkpoint')

    @property
    def server_checkpoint(self):
        return self.get_boolean(self._key, 'server_checkpoint')

    @property
    def daemon_fault_tolerance_cloudlab(self):
        return self.get_property(self._key, 'daemon_fault_tolerance_cloudlab')

    @property
    def daemon_fault_tolerance_bucket(self):
        return self.get_property(self._key, 'daemon_fault_tolerance_bucket')

    @property
    def extra_vm(self):
        return self.get_boolean(self._key, 'extra_vm')

    @property
    def frequency_ckpt(self):
        return self.get_property(self._key, 'frequency_ckpt')

    @property
    def client_checkpoint(self):
        return self.get_boolean(self._key, 'client_checkpoint')

    @property
    def ckpt_file(self):
        return self.get_property(self._key, 'ckpt_file')

    @property
    def folder_checkpoints(self):
        return self.get_property(self._key, 'folder_checkpoints')

    @property
    def provider_bucket(self):
        return self.get_property(self._key, 'provider_bucket')
