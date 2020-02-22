import sys
import random

import namesgenerator
from omegaconf import OmegaConf


def random_name():
    nice_name = namesgenerator.get_random_name()
    random_letters = "".join(chr(random.randint(ord("A"), ord("Z"))) for _ in range(4))
    return nice_name + "_" + random_letters


def summarize_hparams(hparams: OmegaConf):
    hparam_list = []
    for key, value in OmegaConf.to_container(hparams, resolve=True).items():
        fullname, shortname = key.split("--")

        if isinstance(value, bool):
            value = "NY"[value]

        hparam_list.append(f"{shortname}{value}")

    return "_".join(hparam_list)


def parse_args():
    OmegaConf.register_resolver("random_seed", lambda: random.randint(0, 100))
    OmegaConf.register_resolver("random_name", random_name)

    conf = OmegaConf.create()
    for s in sys.argv[1:]:
        if s.endswith(".yaml"):
            conf.merge_with(OmegaConf.load(s))
        else:
            conf.merge_with_dotlist([s])

    # Make sure everything is resolved
    conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True))
    if "fullname" not in conf:
        conf.fullname = "_".join((summarize_hparams(conf.hparams), random_name()))

    return conf
