from control.config.config import Config


class CommunicationConfig(Config):
    _key = 'communication'

    @property
    def ssh_port(self):
        return int(self.get_property(self._key, 'ssh_port'))

    @property
    def repeat(self):
        return int(self.get_property(self._key, 'repeat'))

    @property
    def connection_timeout(self):
        return float(self.get_property(self._key, 'connection_timeout'))

    @property
    def retry_interval(self):
        return int(self.get_property(self._key, 'retry_interval'))

    @property
    def socket_port(self):
        return int(self.get_property(self._key, 'socket_port'))



