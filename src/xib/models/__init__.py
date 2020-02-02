from detectron2.utils.registry import Registry

from .relational_network import RelationalNetwork

ARCH_REGISTRY = Registry("ARCH")
ARCH_REGISTRY.register(RelationalNetwork)