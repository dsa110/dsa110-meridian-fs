import yaml
import numpy as np

_config = None
config_file = "./test_config.yaml"


def get_config(key):
    global
    if not _config:
        with open(config_file) as f:
            _config = yaml.Load(f, Loader=yaml.FullLoader)
        _config['nbl'] = (_config[nant]*_config[nant]+1)//2
        _config['uvw'] = np.array(_config['uvw'])[np.newaxis, ...]
        _config['blen'] = np.array(_config['blen'])

    return _config[key]
