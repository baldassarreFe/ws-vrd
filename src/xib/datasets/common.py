from pathlib import Path
from typing import Union

from PIL import Image

from xib.structures import ImageSize


def get_exif_orientation(img: Image) -> Union[int, None]:
    orientation = img.getexif().get(274)
    return {
        2: "FLIP_LEFT_RIGHT",
        3: "ROTATE_180",
        4: "FLIP_TOP_BOTTOM",
        5: "TRANSPOSE",
        6: "ROTATE_270",
        7: "TRANSVERSE",
        8: "ROTATE_90",
    }.get(orientation)


def img_size_with_exif(img_path: Path):
    with Image.open(img_path.as_posix()) as img:
        width, height = img.size
        exif_orientation = get_exif_orientation(img)
    return ImageSize(height, width), exif_orientation
