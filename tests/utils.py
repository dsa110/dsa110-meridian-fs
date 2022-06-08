import yaml
import numpy as np

_config = None
config_file = "./test_config.yaml"


def get_config(key):
    global _config
    if not _config:
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['nbl'] = (config['nant'] * config['nant'] + 1) // 2
        config['uvw'] = np.array(config['uvw'])[np.newaxis, ...]
        config['blen'] = np.array(config['blen'])

        _config = config

    return _config[key]
