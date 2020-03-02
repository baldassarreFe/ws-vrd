# Datasets

Datasets are downloaded in a `./data` directory, organized as such:
```
data
├── hico_20160224_det
│   ├── raw
│   └── processed
├── vrd
│   ├── raw
│   └── processed
└── README.md
```

## HICO-DET
- [Humans Interacting with Common Objects - Detection](http://www-personal.umich.edu/~ywchao/hico/)
- 38118 training images
- 9658 test images
- Vocabulary
  - 1 subject (person)
  - 117 predicates
  - 80 objects (same as [COCO](http://cocodataset.org/#download))
- Images and annotations (matlab file) 7.9 GB
  [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
  
Download:  
```bash
cd data/hico_20160224_det/raw

# Download hico_20160224_det.tar.gz from Google Drive using a browser 
md5sum hico_20160224_det.tar.gz
# 6cfaeae39b29ecf51f4354ee8ffeef75  hico_20160224_det.tar.gz

tar xvf hico_20160224_det.tar.gz --strip 1
```

Quick test detectron detections:
```bash
cat << 'HERE' > /tmp/d2.sh
#!/usr/bin/env bash
python "${HOME}/detectron2/demo/demo.py" \
--config-file "${HOME}/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" \
--input "$@" \
--output "d2_outputs" \
--opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
HERE

mkdir d2_outputs
chmod u+x /tmp/d2.sh
find "data/hico_20160224_det/raw/images/train2015/" -name '*.jpg' -exec /tmp/d2.sh {} +
rm /tmp/d2.sh
```

Evaluate detectron performances (see this [notebook](../notebooks/COCOeval.ipynb])):
```bash
python -m xib.preprocessing.train_detectron \
  --eval-only \
  --num-gpus=1 \
  --dataset="hico" \
  --data-root="./data" \
  OUTPUT_DIR "models/detectron_hico_eval_only"
```

Preprocess graphs for training:
```bash
python -m xib.preprocessing.hico_det \
  --skip-existing \
  --confidence-threshold=.3 \
  --nms-threshold=.7 \
  --hico-dir="./data" \
  --output-dir="data/hico_20160224_det/processed"
```


## Scene Graphs Dataset
- [Image Retrieval using Scene Graphs](https://hci.stanford.edu/publications/2015/scenegraphs/JohnsonCVPR2015.pdf)
- 5000 images from COCO and YFCC100m
- Open vocabulary:
  - 6000+ object categories, 266 used in the experiment
  - 1300+ triplet categories, 68 used in the experiment
- Download:
  - Images and annotations 1.9 GB 
    [here](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip)

## Visual Relationship Detection Dataset
- [Visual Relationship Detection](https://cs.stanford.edu/people/ranjaykrishna/vrd/)
- Same 5000 images of Scene Graphs:
  - 4006 train
  - 1001
  - Some images have problems with EXIF tags
  - Some other have 0 annotated objects and 0 annotated relations
- Vocabulary (clean version of Scene Graphs):
  - 100 object categories
  - 70 predicates
- Download:
  - Images 1.9 GB 
    [here](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip)
  - Annotations 624K 
    [here](http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip)

Download:    
```bash
cd data/vrd

wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
wget http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip
md5sum sg_dataset.zip json_dataset.zip
# f2ee909ebe04855b2ce8bc9ba4c96a23  sg_dataset.zip
# 654e81f3c581e2cd929567268da8ef66  json_dataset.zip

unzip sg_dataset.zip
rm sg_dataset/sg_{train,test}_annotations.json
mv sg_dataset/* raw
rmdir sg_dataset
unzip -d raw json_dataset.zip
```

Fine-tune detectron and evaluate performances (see this [notebook](../notebooks/COCOeval.ipynb])):
```bash
python -m xib.preprocessing.train_detectron \
  --num-gpus=2 \
  --dataset="vrd" \
  --data-root="./data" \
  OUTPUT_DIR "models/detectron_vrd_train" \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.MAX_ITER 12000
```

Preprocess graphs for training:
```bash
python -m xib.preprocessing.vrd \
  --confidence-threshold=.3 \
  --d2-dir="models/detectron_vrd_train" \
  --data-dir="./data" \
  --output-dir="data/vrd/processed"
```

## UnRel
- [Unusual Relations](https://www.di.ens.fr/willow/research/unrel/)
- 1000+ images
- 76 triplets
- Vocabulary (same as Visual Relationship Detection)
  - 100 subjects
  - 70 predicates
  - 100 objects
- 213 Mb (in matlab)

Download
```bash
cd data/unrel/raw

wget https://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz
md5sum unrel-dataset.tar.gz    
# c6c2b20f3b4c6b5a85f070898fa5b5c1  unrel-dataset.tar.gz

tar xvf unrel-dataset.tar.gz
```

Preprocess graphs for training:
```bash
python -m xib.preprocessing.vrd \
  --confidence-threshold=.3 \
  --d2-dir="models/detectron_vrd_train" \
  --data-dir="./data" \
  --output-dir="data/vrd/processed"
```

To evaluate the model on UnRel with VRD as a confounder (called `unrel_vrd` in the dataset catalog), set up these links:
```bash
cd data
mkdir -p unrel_vrd/raw/images unrel_vrd/processed/test

for f in unrel/raw/images/*.jpg; do
    ln -s "$(realpath "$f")" "unrel_vrd/raw/images/$(basename "$f")"
done

for f in vrd/raw/sg_test_images/*.jpg; do
    ln -s "$(realpath "$f")" "unrel_vrd/raw/images/$(basename "$f")"
done

for f in unrel/processed/test/*.graph.pth; do
    ln -s "$(realpath "$f")" "unrel_vrd/processed/test/$(basename "$f")"
done

for f in vrd/processed/test/*.graph.pth; do
    ln -s "$(realpath "$f")" "unrel_vrd/processed/test/$(basename "$f")"
done
```

## COCO-a
- Subset of COCO
- 4413 images
- Vocabulary
  - 80 subjects
  - 140 predicates
  - 80 objects