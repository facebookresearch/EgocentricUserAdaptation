#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import logging
import os
import sys
from continual_ego4d.utils.misc import makedirs

from . import distributed as du


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def setup_logging(output_dirs=None, host_name=None, overwrite_logfile=False):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    _logger = logging.getLogger("lightning")
    _logger.handlers = []

    logger = logging.getLogger()
    logger.handlers = []  # Reset
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    process_info = f"[{host_name}]" if host_name is not None else ""
    plain_formatter = logging.Formatter(
        f"{process_info}[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    # Suppress: Only main process outputs to stdout (rank 0)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)  # Output to STDOUT
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    # Redirect ALL stderr
    # Directly set stream output, as exceptions result in termination before anything is logged.
    # sys.stdout = StreamToLogger(logger, logging.INFO) # Don't redirect as gives
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # Can add multiple output logfiles (e.g. host-specific by rank)
    if output_dirs is not None:
        if not isinstance(output_dirs, list):
            output_dirs = [output_dirs]
        for output_dir in output_dirs:
            makedirs(output_dir, exist_ok=True)
            log_filename = "stdout_{}{}.log"
            if overwrite_logfile:
                filename = os.path.join(output_dir, log_filename.format('host', du.get_rank()))
            else:
                counter = 0
                filename = os.path.join(output_dir, log_filename.format('v', counter))
                while os.path.exists(filename):
                    counter += 1
                    filename = os.path.join(output_dir, log_filename.format('v', counter))

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
