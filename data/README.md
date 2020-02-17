# Datasets

## HICO-DET
- [Humans Interacting with Common Objects - Detection](http://www-personal.umich.edu/~ywchao/hico/)
- 38118 training images
- 9658 test images
- Vocabulary
  - 1 subject (person)
  - 117 predicates
  - 80 objects (same as [COCO](http://cocodataset.org/#download))
- 7.9 GB (in matlab)

Test detectron detections with:
```bash
cat << 'HERE' > /tmp/d2.sh
#!/usr/bin/env bash
cd "${HOME}/detectron2"
mkdir -p "${HOME}/Desktop/eccv/data/hico_20160224_det/images/train2015_bb/"
python demo/demo.py \
--config-file configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
--input "$@" \
--output "${HOME}/Desktop/eccv/data/hico_20160224_det/images/train2015_bb/" \
--opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
HERE
chmod u+x /tmp/d2.sh
find "${HOME}/Desktop/eccv/data/hico_20160224_det/images/train2015/" -name '*.jpg' -exec /tmp/d2.sh {} +
rm /tmp/d2.sh
```

Preprocess for training with:
```bash
python -m xib.preprocessing.hico_det \
  --hico-dir="data/raw/hico_20160224_det" \
  --skip-existing \
  --confidence-threshold=.3 \
  --nms-threshold=.7 \
  --output-dir="data/processed/hico_20160224_det"
```

## Visual Relationship Detection
- [Visual Relationship Detection](https://cs.stanford.edu/people/ranjaykrishna/vrd/)
- 5000+ images, same of [Scene Graphs](https://cs.stanford.edu/~danfei/scene-graph/)
- Vocabulary
  - 100 subjects
  - 70 predicates
  - 100 objects
- Download:
  - Images 1.9 GB [here](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip)
  - Annotations [here](http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip)

## UnRel
- [Unusual Relations](https://www.di.ens.fr/willow/research/unrel/)
- 1000+ images
- 76 triplets
- Vocabulary (same as Visual Relationship Detection)
  - 100 subjects
  - 70 predicates
  - 100 objects
- 213 Mb (in matlab)

## COCO-a
- Subset of COCO
- 4413 images
- Vocabulary
  - 80 subjects
  - 140 predicates
  - 80 objects