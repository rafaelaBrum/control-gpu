import json
import logging
import time

import xmltodict

import control.managers.cloudlab_rpc as prpc

from control.util.loader import Loader

from control.domain.instance_type import InstanceType

from typing import List

import string
import random


def _random_string(str_len):
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(str_len))


class Node:
    """Represents a node on the Powder platform. Holds an SSHConnection instance for
    interacting with the node.

    Attributes:
        client_id (str): Matches the id defined for the node in the Powder profile.
        ip_address (str): The public IP address of the node.
        hostname (str): The hostname of the node.

    """
    def __init__(self, client_id, ip_address, hostname, instance_type, loader, market):
        super().__init__(instance_type, market, loader)
        self.client_id = client_id
        self.ip_address = ip_address
        self.hostname = hostname


class Experiment:
    """Represents a single experiment. Can be used to start, interact with,
    and terminate the experiment. After an experiment is ready, this object
    holds references to the nodes in the experiment.

    Args:
        experiment_name (str): A name for the experiment. Must be less than 16 characters.
        profile_name (str): The name of an existing CloudLab profile you want to use for the experiment.

    Attributes:
        status (int): Represents the last known status of the experiment as
            retrieved from the Powder RPC server.
        nodes (dict of str: Node): A lookup table mapping node ids to Node instances
            in the experiment.
        experiment_name (str)
        project_name (str)
        profile_name (str)

    """

    MARKET = 'emulated'

    EXPERIMENT_NOT_STARTED = 0
    EXPERIMENT_PROVISIONING = 1
    EXPERIMENT_PROVISIONED = 2
    EXPERIMENT_READY = 3
    EXPERIMENT_FAILED = 4
    EXPERIMENT_NULL = 5
    EXPERIMENT_STARTED = 6

    POLL_INTERVAL_S = 20
    PROVISION_TIMEOUT_S = 1800
    MAX_NAME_LENGTH = 16

    EXPERIMENT_RANDOM = _random_string(2)

    CLUSTER_URN_DEFAULT = 'urn:publicid:IDN+utah.cloudlab.us+authority+cm'

    def __init__(self, experiment_name, loader: Loader, profile_name, cluster=None, bindings=None,
                 instances_types: List[InstanceType] = None):

        experiment_name = experiment_name + self.EXPERIMENT_RANDOM
        if len(experiment_name) > 16:
            logging.error('<Experiment CloudLab Manager> Experiment name {} is too long '
                          '(cannot exceed {} characters)'.format(experiment_name,
                                                                 self.MAX_NAME_LENGTH))

        self.valid = not (len(experiment_name) > 16)
        self.loader = loader
        self.experiment_name = experiment_name
        self.project_name = loader.cloudlab_conf.project_name
        self.profile_name = profile_name
        self.instance_types = instances_types
        if cluster is None:
            self.cluster_urn = self.CLUSTER_URN_DEFAULT
        else:
            self.cluster_urn = cluster
        self.bindings = bindings
        self.status = self.EXPERIMENT_NOT_STARTED
        self.nodes = dict()
        self._manifests = None
        self._poll_count_max = self.PROVISION_TIMEOUT_S // self.POLL_INTERVAL_S
        logging.info('<Experiment CloudLab Manager> initialized experiment {} based on profile {} '
                     'under project {} on cluster {} with bindings {}'.format(self.experiment_name,
                                                                              self.profile_name,
                                                                              self.project_name,
                                                                              self.cluster_urn,
                                                                              self.bindings))

    def start_and_wait(self):
        """Start the experiment and wait for READY or FAILED status."""
        logging.info('<Experiment CloudLab Manager> starting experiment {} in cluster {}'.format(self.experiment_name,
                                                                                                 self.cluster_urn))
        return_val, response = prpc.start_experiment(self.experiment_name,
                                                     self.project_name,
                                                     self.profile_name, self.cluster_urn,
                                                     self.bindings)
        if return_val == prpc.RESPONSE_SUCCESS:
            self._get_status()

            poll_count = 0
            while self.still_provisioning and poll_count < self._poll_count_max:
                self._get_status()
                time.sleep(self.POLL_INTERVAL_S)
        else:
            self.status = self.EXPERIMENT_FAILED
            logging.info(response)

        return self.status

    def terminate(self):
        """Terminate the experiment. All allocated resources will be released."""
        logging.info('<Experiment CloudLab Manager> terminating experiment {}'.format(self.experiment_name))
        return_val, response = prpc.terminate_experiment(self.project_name, self.experiment_name)
        if return_val == prpc.RESPONSE_SUCCESS:
            self.status = self.EXPERIMENT_NULL
        else:
            logging.error(f'<Experiment CloudLab Manager> Experiment {self.experiment_name}: '
                          f'failed to terminate experiment')
            logging.error('<Experiment CloudLab Manager> Experiment {}: output {}'.format(self.experiment_name,
                                                                                          response['output']))

        return self.status

    def _get_manifests(self):
        """Get experiment manifests, translate to list of dicts."""
        return_val, response = prpc.get_experiment_manifests(self.project_name,
                                                             self.experiment_name)
        if return_val == prpc.RESPONSE_SUCCESS:
            response_json = json.loads(response['output'])
            self._manifests = [xmltodict.parse(response_json[key]) for key in response_json.keys()]
            logging.info(f'<Experiment CloudLab Manager> Experiment {self.experiment_name}: got manifests')
        else:
            logging.error(f'<Experiment CloudLab Manager> Experiment {self.experiment_name}: failed to get manifests')

        return self

    def _parse_manifests(self):
        """Parse experiment manifests and add nodes to lookup table."""
        # print(json.dumps(self._manifests, indent=4))
        for manifest in self._manifests:
            # print("manifest['rspec']")
            # print(json.dumps(manifest['rspec'], indent=4))
            nodes = manifest['rspec']['node']
            # print("nodes")
            # print(json.dumps(nodes, indent=4))
            single_node = False
            index = 0
            for node in nodes:
                if isinstance(node, str):
                    single_node = True
                    break
                # print("node")
                # print(json.dumps(node, indent=4))
                try:
                    hostname = node['host']['@name']
                    ipv4 = node['host']['@ipv4']
                    client_id = node['@client_id']
                    self.nodes[client_id] = Node(client_id=client_id, ip_address=ipv4, hostname=hostname,
                                                 instance_type=self.instance_types[index], loader=self.loader,
                                                 market=self.MARKET)
                    logging.info('<Experiment CloudLab Manager> Experiment {}: '
                                 'parsed manifests for node {}'.format(self.experiment_name, client_id))
                    index = index+1
                except Exception as e:
                    logging.error(f"<Experiment CloudLab Manager> Experiment {self.experiment_name}: "
                                  f"Error trying to parse manifests for node {node['@client_id']}")
                    logging.error(e)
                    pass
            if single_node:
                try:
                    hostname = nodes['host']['@name']
                    ipv4 = nodes['host']['@ipv4']
                    client_id = nodes['@client_id']
                    self.nodes[client_id] = Node(client_id=client_id, ip_address=ipv4, hostname=hostname,
                                                 instance_type=self.instance_types[0], loader=self.loader,
                                                 market=self.MARKET)
                    logging.info('<Experiment CloudLab Manager> Experiment {}: '
                                 'parsed manifests for node {}'.format(self.experiment_name, client_id))
                except Exception as e:
                    logging.error(f"<Experiment CloudLab Manager> Experiment {self.experiment_name}: "
                                  f"Error trying to parse manifests for node {nodes['@client_id']}")
                    logging.error(e)
                    pass

        return self

    def _get_status(self):
        """Get experiment status and update local state. If the experiment is ready, get
        and parse the associated manifests.

        """
        return_val, response = prpc.get_experiment_status(self.project_name,
                                                          self.experiment_name)
        if return_val == prpc.RESPONSE_SUCCESS:
            output = response['output']
            if 'Status: ready\n' in output:
                self.status = self.EXPERIMENT_READY
                self._get_manifests()._parse_manifests()
            elif 'Status: provisioning\n' in output:
                self.status = self.EXPERIMENT_PROVISIONING
            elif 'Status: provisioned\n' in output:
                self.status = self.EXPERIMENT_PROVISIONED
            elif 'Status: failed\n' in output:
                self.status = self.EXPERIMENT_FAILED
            elif 'Status: created\n' in output:
                self.status = self.EXPERIMENT_STARTED

            self.still_provisioning = self.status in [self.EXPERIMENT_PROVISIONING,
                                                      self.EXPERIMENT_PROVISIONED,
                                                      self.EXPERIMENT_STARTED]
            logging.info('<Experiment CloudLab Manager> Experiment {}: '
                         'experiment status is {} in cluster {}'.format(self.experiment_name,
                                                                        output.split('\n')[0],
                                                                        self.cluster_urn))
        else:
            logging.error(f'<Experiment CloudLab Manager> Experiment {self.experiment_name}: '
                          f'failed to get experiment status')

        return self
