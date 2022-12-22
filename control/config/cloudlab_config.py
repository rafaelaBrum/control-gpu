from control.config.config import Config


class CloudLabConfig(Config):
    _key = 'cloudlab'

    @property
    def project_name(self):
        return self.get_property(self._key, 'project_name')

    @property
    def server_experiment_name(self):
        return self.get_property(self._key, 'server_experiment_name')

    @property
    def client_experiment_name(self):
        return self.get_property(self._key, 'client_experiment_name')

    @property
    def key_path(self):
        return self.get_property(self._key, 'key_path')

    @property
    def vm_user(self):
        return self.get_property(self._key, 'vm_user')

    @property
    def home_path(self):
        return self.get_property(self._key, 'home_path')

    @property
    def cpu_costs(self):
        return self.get_property(self._key, 'cpu_costs')

    @property
    def ram_costs(self):
        return self.get_property(self._key, 'ram_costs')

    @property
    def gpu_v100_costs(self):
        return self.get_property(self._key, 'gpu_v100_costs')

    @property
    def gpu_p100_costs(self):
        return self.get_property(self._key, 'gpu_p100_costs')

    @property
    def gpu_k40_costs(self):
        return self.get_property(self._key, 'gpu_k40_costs')

    @property
    def extra_ds_path(self):
        return self.get_property(self._key, 'extra_ds_path')
