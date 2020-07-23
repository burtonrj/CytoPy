import os

import mongoengine


def _valid_directory(val: str):
    if not os.path.isdir(val):
        raise mongoengine.errors.ValidationError(f"{val} is not a valid directory")