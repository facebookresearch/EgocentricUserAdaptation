#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Method construction functions."""

from fvcore.common.registry import Registry

METHOD_REGISTRY = Registry("METHOD")
METHOD_REGISTRY.__doc__ = """
Registry for Method Callback.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_method(cfg, trainer, name=None):
    """
    Builds the method callback.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        method. Details can be seen in ego4d/config/defaults.py.
    """
    # Construct the method
    name = cfg.METHOD.METHOD_NAME if name is None else name
    method = METHOD_REGISTRY.get(name)(cfg, trainer)
    return method
