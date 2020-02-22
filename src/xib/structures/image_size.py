from typing import NamedTuple


class ImageSize(NamedTuple):
    height: int
    width: int

    @property
    def area(self):
        return self.height * self.width

    def __str__(self):
        return f"(h={self.height}, w={self.width})"
