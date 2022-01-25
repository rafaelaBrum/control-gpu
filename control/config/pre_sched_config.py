from control.config.config import Config


class PreSchedConfig(Config):
    _key = 'pre_sched'

    @property
    def path(self):
        return self.get_property(self._key, 'path')

    @property
    def rtt_file(self):
        return self.get_property(self._key, 'rtt_file')

    @property
    def train_file(self):
        return self.get_property(self._key, 'train_file')

    @property
    def app_file(self):
        return self.get_property(self._key, 'app_file')

    @property
    def results_temp_file(self):
        return self.get_property(self._key, 'results_temp_file')
