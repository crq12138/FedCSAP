import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import hashlib
import pickle
import yaml

def seed_from(base_seed, *components):
    payload = f'{int(base_seed)}::' + '::'.join(str(c) for c in components)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)

def set_global_seeds(seed, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


def get_hash_from_param_file(param):
    if 'hash' not in param:
        hash_md5 = hashlib.md5()

        param = dict(param)
        
        # generate hash from yaml file
        hash_md5.update(pickle.dumps(param))
        return hash_md5.hexdigest()
    else:
        return param['hash']


if __name__ == '__main__':
    # load yaml file
    with open(f'utils/cifar_params.yaml', 'r') as f:
        param = yaml.safe_load(f)

    print(get_hash_from_param_file(param))
