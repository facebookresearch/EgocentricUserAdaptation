""" This script should have no dependencies on other ego4d/continual_ego4d to avoid circular imports."""
import os

def makedirs(path, mode=0o777, exist_ok=True):
    """Fix to change umask in order for makedirs to work. """
    try:
        original_umask = os.umask(0)
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
    finally:
        os.umask(original_umask)