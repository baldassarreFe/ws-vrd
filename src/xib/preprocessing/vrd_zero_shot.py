from pathlib import Path

from detectron2.data import MetadataCatalog

from xib.datasets import register_vrd
from xib.datasets.vrd.catalog import get_zero_shot_data_dicts, register_vrd_zero_shot

# Once preprocessing of VRD is done (images and graphs are in data/vrd)
# run this script to populate data/vrd_zero_shot with symlinks

register_vrd("./data")
register_vrd_zero_shot("./data")

vrd_test_meta = MetadataCatalog.get("vrd_relationship_detection_test")
vrd_zero_shot_meta = MetadataCatalog.get("vrd_relationship_detection_zero_shot")

Path(vrd_zero_shot_meta.image_root).mkdir(parents=True, exist_ok=True)
Path(vrd_zero_shot_meta.graph_root).mkdir(parents=True, exist_ok=True)

zero_shot_images = 0
for d in get_zero_shot_data_dicts():
    img_test = Path(d["file_name"])
    img_zero_shot = Path(vrd_zero_shot_meta.image_root) / Path(d["file_name"]).name
    if img_zero_shot.exists():
        if img_zero_shot.is_symlink():
            img_zero_shot.unlink()
        else:
            raise RuntimeError(f"{img_zero_shot} exists and is not a symlink")
    img_zero_shot.symlink_to(img_test)

    pth_test = (
        Path(vrd_test_meta.graph_root)
        / Path(d["file_name"]).with_suffix(".graph.pth").name
    )
    pth_zero_shot = (
        Path(vrd_zero_shot_meta.graph_root)
        / Path(d["file_name"]).with_suffix(".graph.pth").name
    )
    if pth_zero_shot.exists():
        if pth_zero_shot.is_symlink():
            pth_zero_shot.unlink()
        else:
            raise RuntimeError(f"{pth_zero_shot} exists and is not a symlink")
    pth_zero_shot.symlink_to(pth_test)

    zero_shot_images += 1

print(f"Linked {zero_shot_images} zero-shot samples to data/vrd_zero_shot")
