from control.config.config import Config


class SimulationConfig(Config):
    _key = 'simulation'

    @property
    def with_simulation(self):
        return self.get_boolean(self._key, 'with_simulation')

    @property
    def faulty_server(self):
        return self.get_boolean(self._key, 'faulty_server')

    @property
    def faulty_clients(self):
        return self.get_boolean(self._key, 'faulty_clients')

    @property
    def revocation_rate(self):
        return float(self.get_property(self._key, 'revocation_rate'))

