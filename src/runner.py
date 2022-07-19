import os
import yaml


class RLRunner:
    def __init__(self, path_config=None):
        def read_config(conf):
            with open(conf, 'r') as stream:
                try:
                    config_ = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            return config_

        self.config = read_config(path_config)
        print(self.config)


if __name__ == "__main__":
    RLRunner("../experiment.yml")
