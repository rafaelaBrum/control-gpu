from control.config.config import Config


class SimulationConfig(Config):
    _key = 'simulation'

    @property
    def with_simulation(self):
        return self.get_boolean(self._key, 'with_simulation')

    @property
    def revocation_rate(self):
        return float(self.get_property(self._key, 'revocation_rate'))

