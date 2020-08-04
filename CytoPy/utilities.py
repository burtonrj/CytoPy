import os
import mongoengine


def valid_directory(val: str):
    if not os.path.isdir(val):
        raise mongoengine.errors.ValidationError(f"{val} is not a valid directory")


def indexed_parallel_func(x: tuple,
                          func: callable,
                          **kwargs):
    x[0], func(x[1].values, **kwargs)