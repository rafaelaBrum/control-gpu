from control.config.config import Config


class PreSchedConfig(Config):
    _key = 'pre_sched'

    @property
    def rtt_path(self):
        return self.get_property(self._key, 'rtt_path')

    @property
    def rtt_file(self):
        return self.get_property(self._key, 'rtt_file')
