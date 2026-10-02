import os

import yaml


class Config:
    def __init__(self, root_dir=None, yaml_file=None):

        self._config_dict = {}
        if root_dir:
            yaml_path = os.path.join(root_dir, 'configs', yaml_file)
            self.load_from_yaml(yaml_path)

    def load_from_yaml(self, yaml_path):

        with open(yaml_path, 'r', encoding='utf-8') as f:
            self._config_dict = yaml.safe_load(f)

    def get(self, key, default=None):
        return self._config_dict.get(key, default)

    def update(self, key, value):

        self._config_dict[key] = value

    def __getattr__(self, key):
        if key in self._config_dict:
            value = self._config_dict[key]
            if isinstance(value, dict):
                wrapped = Config.from_dict(value)
                self._config_dict[key] = wrapped
                return wrapped
            else:
                return value
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    @classmethod
    def from_dict(cls, d):
        config = cls()
        config._config_dict = d
        return config

    def __repr__(self):
        return f"Config({self._config_dict})"
