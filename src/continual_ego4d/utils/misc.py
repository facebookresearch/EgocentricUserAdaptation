# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" This script should have no dependencies on other ego4d/continual_ego4d to avoid circular imports."""
import os
from fvcore.common.config import CfgNode


def makedirs(path, mode=0o777, exist_ok=True):
    """Fix to change umask in order for makedirs to work. """
    try:
        original_umask = os.umask(0)
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
    finally:
        os.umask(original_umask)


class SoftCfgNode(CfgNode):
    """ Don't generate exceptions on replacing attributes. """

    def __setattr__(self, name: str, val) -> None:  # pyre-ignore
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                print(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(name, old_val, val)
                )
            self[name] = val
        else:
            super().__setattr__(name, val)
