from detectron2.structures import Instances


def clone_instances(instances: Instances):
    return Instances(
        instances.image_size,
        **{
            k: v.clone()
            for k, v in instances.get_fields().items()
        }
    )