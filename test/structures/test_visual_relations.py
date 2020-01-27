import torch

from xib.structures import VisualRelations, Boxes


def test_constructor(device):
    vr = VisualRelations(
        subject_classes=torch.tensor([1, 2, 3], device=device),
        predicate_classes=torch.tensor([1, 2, 3], device=device),
        object_classes=torch.tensor([1, 2, 3], device=device),

        subject_boxes=Boxes(torch.rand(3, 4, device=device)),
        object_boxes=Boxes(torch.rand(3, 4, device=device)),
    )

    assert vr.device.type == device
    assert len(vr) == 3
