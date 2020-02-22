import importlib
import inspect

from loguru import logger


def import_(fullname):
    package, name = fullname.rsplit(".", maxsplit=1)
    package = importlib.import_module(package)
    return getattr(package, name)


def check_extra_parameters(f, kwargs):
    signature = inspect.signature(f)
    if not set(kwargs.keys()).issubset(set(signature.parameters.keys())):
        logger.warning(
            f"Extra parameters found for {f}, "
            f"expected {{{set(signature.parameters.keys())}}}, "
            f"given {{{set(kwargs.keys())}}}"
        )
