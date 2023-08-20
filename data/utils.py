import yaml


def load_args(config_path):
    with open(config_path) as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_obj