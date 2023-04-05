from control.config.config import Config


class InputConfig(Config):
    _key = 'input'

    @property
    def path(self):
        return self.get_property(self._key, 'path')

    @property
    def job_file(self):
        return self.get_property(self._key, 'job_file')

    @property
    def env_file(self):
        return self.get_property(self._key, 'env_file')

    @property
    def loc_file(self):
        return self.get_property(self._key, 'loc_file')

    @property
    def pre_file(self):
        return self.get_property(self._key, 'pre_file')

    @property
    def input_file(self):
        return self.get_property(self._key, 'input_file')

    @property
    def map_file(self):
        return self.get_property(self._key, 'map_file')

    @property
    def deadline_seconds(self):
        return float(self.get_property(self._key, 'deadline_seconds'))

    @property
    def task_leaf(self):
        return self.get_boolean(self._key, 'task_leaf')

    # @property
    # def ac_size_seconds(self):
    #     return float(self.get_property(self._key, 'ac_size_seconds'))

    # @property
    # def idle_slack_time(self):
    #     return float(self.get_property(self._key, 'idle_slack_time'))
