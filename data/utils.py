import os
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





