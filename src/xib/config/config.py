import sys
import random

import namesgenerator
from omegaconf import OmegaConf


def random_name():
    nice_name = namesgenerator.get_random_name()
    random_letters = ''.join(chr(random.randint(ord('A'), ord('Z'))) for _ in range(6))
    return nice_name + '_' + random_letters


def parse_args():
    OmegaConf.register_resolver('randomname', random_name)

    conf = OmegaConf.create()
    for s in sys.argv[1:]:
        if s.endswith('.yaml'):
            conf.merge_with(OmegaConf.load(s))
        else:
            conf.merge_with_dotlist([s])

    # Make sure everything is resolved
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    return conf
