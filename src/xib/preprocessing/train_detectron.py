import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import verify_results
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.utils import comm as comm
from loguru import logger


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """

    if args.dataset == "vrd":
        from xib.datasets.vrd import register_vrd

        register_vrd(args.data_root)
    elif args.dataset == "hico":
        from xib.datasets.hico_det import register_hico

        register_hico(args.data_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    cfg = get_cfg()
    cfg.merge_from_file(
        get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )

    cfg.MODEL.WEIGHTS = get_checkpoint_url(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = (f"{args.dataset}_object_detection_train",)
    cfg.DATASETS.TEST = (f"{args.dataset}_object_detection_test",)

    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        MetadataCatalog.get(f"{args.dataset}_object_detection_train").thing_classes
    )

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


"""
# Example:
python -m xib.preprocessing.train_detectron \
  --dataset=vrd \
  --data-root=data/raw/vrd \
  --eval-only
"""
if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Which dataset to use.",
        choices=["hico", "vrd"],
    )
    parser.add_argument(
        "--data-root", required=True, help="Where the raw dataset is stored."
    )
    args = parser.parse_args()

    if args.config_file != "":
        logger.warning(f"Config file will be ignored: {args.config_file}")
    del args.config_file

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
