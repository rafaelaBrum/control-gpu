from control.config.config import Config


class CheckPointConfig(Config):
    _key = 'checkpoint'

    @property
    def with_checkpoint(self):
        return self.get_boolean(self._key, 'with_checkpoint')

    @property
    def daemon_fault_tolerance(self):
        return self.get_property(self._key, 'daemon_fault_tolerance')

    @property
    def extra_vm(self):
        return self.get_boolean(self._key, 'extra_vm')

    @property
    def frequency_ckpt(self):
        return self.get_property(self._key, 'frequency_ckpt')
