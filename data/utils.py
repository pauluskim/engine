import os
import time
from functools import wraps
from pathlib import Path

import yaml


def load_args(config_path):
    with open(config_path) as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_obj

def mkdir_if_not_exist(path):
    def has_extension(path):
        return "." in path.split("/")[-1]

    if has_extension(path):  # path is for a file.
        path = "/".join(path.split("/")[:-1])

    if not Path(path).exists():
        os.makedirs(path)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"Function {func.__name__}{args} {kwargs} took {elapsed:.2f} ms")
        return result
    return timeit_wrapper





