#!/usr/bin/env python3
#
# Copyright (c) 2004-2020 University of Utah and the Flux Group.
#
# {{{EMULAB-LICENSE
#
# This file is part of the Emulab network testbed software.
#
# This file is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.
#
# }}}
#

import logging
import ssl
import sys
import xmlrpc.client as xmlrpc_client

import json

from pathlib import Path

PACKAGE_VERSION = 0.1
DEBUG = 0

# rpc server
XMLRPC_SERVER = "boss.emulab.net"
XMLRPC_PORT = 3069
SERVER_PATH = "/usr/testbed"
URI = "https://" + XMLRPC_SERVER + ":" + str(XMLRPC_PORT) + SERVER_PATH

RESPONSE_SUCCESS = 0
RESPONSE_BAD_ARGS = 1
RESPONSE_ERROR = 2
RESPONSE_FORBIDDEN = 3
RESPONSE_BAD_VERSION = 4
RESPONSE_SERVERERROR = 5
RESPONSE_TOO_BIG = 6
RESPONSE_REFUSED = 7  # Emulab is down, try again later.
RESPONSE_TIMED_OUT = 8

JSON_FILE_CRED = Path(Path.home(), "cloudlab_entry_fields.json")

# User supplied login ID, password, and certificate
try:
    with open(JSON_FILE_CRED) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    LOGIN_ID = jsonObject['USER']
    PEM_PWORD = jsonObject['PWORD']
    CERT_PATH = jsonObject['CERT']
except KeyError:
    logging.error('<CloudLab RPC> Missing CloudLab credential environment variable(s)')
    sys.exit(1)


def do_method(method, params):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.load_cert_chain(CERT_PATH, password=PEM_PWORD)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Get a handle on the server,
    server = xmlrpc_client.ServerProxy(URI, context=ctx, verbose=DEBUG)

    # Get a pointer to the function we want to invoke.
    meth = getattr(server, "portal." + method)
    meth_args = [PACKAGE_VERSION, params]

    # Make the call.
    try:
        response = meth(*meth_args)
    except xmlrpc_client.Fault as e:
        print(e.faultString)
        return -1, None

    return_val = response["code"]

    # If the code indicates failure, look for a "value". Use that as the
    # return value instead of the code.
    if return_val != RESPONSE_SUCCESS:
        if response["value"]:
            return_val = response["value"]

    return return_val, response


def start_experiment(experiment_name, project_name, profile_name, cluster_urn, bindings):
    if bindings is None:
        params = {
            "name": experiment_name,
            "proj": project_name,
            "duration": "16",
            "profile": ','.join([project_name, profile_name]),
            "aggregate": cluster_urn
        }
    else:
        params = {
            "name": experiment_name,
            "proj": project_name,
            "duration": "16",
            "profile": ','.join([project_name, profile_name]),
            "aggregate": cluster_urn,
            "bindings": bindings
        }
    # print(params)
    return_val, response = do_method("startExperiment", params)
    return return_val, response


def terminate_experiment(project_name, experiment_name):
    params = {
        "experiment": ','.join([project_name, experiment_name])
    }
    return_val, response = do_method("terminateExperiment", params)
    return return_val, response


def get_experiment_status(project_name, experiment_name):
    params = {
        "experiment": ','.join([project_name, experiment_name])
    }
    return_val, response = do_method("experimentStatus", params)
    return return_val, response


def get_experiment_manifests(project_name, experiment_name):
    params = {
        "experiment": ','.join([project_name, experiment_name])
    }
    return_val, response = do_method("experimentManifests", params)
    return return_val, response
