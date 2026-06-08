# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ProtoBuf serialization and deserialization."""

from typing import Any, List, cast

from .dummy_pb2 import (
    Parameters,
    Scalar,
    FitRes,
    FitIns,
    EvaluateRes,
    EvaluateIns
)

from . import dummy_typing as typing

# pylint: disable=missing-function-docstring

GRPC_MAX_MESSAGE_LENGTH = 1_073_741_824


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None and res.fit_duration is not None:
        return FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None:
        return FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.fit_duration is not None:
        return FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return FitRes(
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: FitRes) -> typing.FitRes:
    """Deserialize flower.FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        parameters=parameters,
        num_examples=msg.num_examples,
        num_examples_ceil=msg.num_examples_ceil,  # Deprecated
        fit_duration=msg.fit_duration,  # Deprecated
        metrics=metrics,
    )


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: EvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> EvaluateRes:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.accuracy is not None:
        return EvaluateRes(
            loss=res.loss,
            num_examples=res.num_examples,
            accuracy=res.accuracy,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return EvaluateRes(
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: EvaluateRes) -> typing.EvaluateRes:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        loss=msg.loss,
        num_examples=msg.num_examples,
        accuracy=msg.accuracy,  # Deprecated
        metrics=metrics,
    )


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize... ."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize... ."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize... ."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize... ."""
    scalar = getattr(scalar_msg, scalar_msg.WhichOneof("scalar"))
    return cast(typing.Scalar, scalar)
