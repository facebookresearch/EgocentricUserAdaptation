#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import logging
import os
import sys

from . import distributed as du


def setup_logging(output_dirs=None, host_name=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    _logger = logging.getLogger("lightning")
    _logger.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    device_info = f"{host_name} " if host_name is not None else ""
    plain_formatter = logging.Formatter(
        f"[{device_info}%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    # Suppress: Only main process (rank 0)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)
    else:
        # For debug, need errors from any of the processes:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(logging.WARNING)  # Warning and up from all processes
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    # Can add multiple output logfiles (e.g. host-specific by rank)
    if output_dirs is not None:
        if not isinstance(output_dirs, list):
            output_dirs = [output_dirs]
        for output_dir in output_dirs:
            filename = os.path.join(output_dir, f"stdout_{du.get_rank()}.log")
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
