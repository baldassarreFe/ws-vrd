from .config import parse_args


def print_config():
    conf = parse_args()
    print(conf.pretty())


if __name__ == "__main__":
    print_config()
