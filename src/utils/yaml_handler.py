from types import SimpleNamespace

import yaml


def load(yaml_path: str) -> SimpleNamespace:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    return _get_reccursive(cfg)


def _get_reccursive(config: dict) -> SimpleNamespace:
    new_config = SimpleNamespace(**config)
    for name, values in new_config.__dict__.items():
        if type(values) is dict:
            new_config.__setattr__(name, _get_reccursive(values))
        else:
            continue
    return new_config
