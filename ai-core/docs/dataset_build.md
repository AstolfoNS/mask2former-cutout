# Cutout Dataset Build Guide

This project should train on compact, real-scene person/car data rather than
the old single-instance HVM-only dataset.

## Target Mix

The default COCO+VOC recipe builds about 6000 images when the source data is
available:

- Up to 3200 images containing both person and car.
- Up to 1400 person-only images.
- Up to 1400 car-only images.
- At least 600 images from Pascal VOC 2012 trainval.

The builder enforces a minimum selected count, a minimum person+car ratio, and
source-level minimums so non-COCO samples cannot be accidentally squeezed out by
COCO-only candidates.

The final training dataset may exist only on the training server. If this WSL
checkout has no `ai-core/data/cutout_mix_512/` directory, rebuild it with the
commands below or copy it from the training machine.

## Source Layout

Place small source subsets here:

```text
ai-core/data/sources/
  coco/
    annotations/
      instances_train2017.json
    train2017/
      *.jpg
  cityscapes/
    leftImg8bit/
      train/
        <city>/*_leftImg8bit.png
    gtFine/
      train/
        <city>/*_gtFine_polygons.json
  voc2012/
    images/
      *.jpg
    masks/
      *.png
    label_map.json
```

COCO should be the first source to add because it provides real person/car
instance masks with the lowest conversion cost. Cityscapes is useful for street
scene hard cases. Pascal VOC 2012 adds non-COCO semantic segmentation samples
for generalization.

## Build

Prepare Pascal VOC 2012 semantic segmentation data first:

```bash
uv run --project ai-core python ai-core/scripts/data/prepare_pascal_voc2012.py
```

Download the compact COCO person/car subset:

```bash
uv run --project ai-core python ai-core/scripts/data/download_and_build_coco_cutout.py \
    --clean-output
```

This command downloads only the COCO annotations plus selected person/car
images. It does not download the full train2017 image archive. The command also
builds a COCO-only dataset, but the recommended final dataset should be rebuilt
from the COCO+VOC config after VOC is prepared.

Use smaller targets for a quick first run:

```bash
uv run --project ai-core python ai-core/scripts/data/download_and_build_coco_cutout.py \
    --output-dir ai-core/data/cutout_mix_512_small \
    --final-total 500 \
    --final-both 280 \
    --final-person-only 90 \
    --final-car-only 90 \
    --min-selected 400 \
    --clean-output
```

Build the recommended COCO+VOC dataset from already downloaded sources:

```bash
uv run --project ai-core python ai-core/scripts/data/build_cutout_dataset.py \
    --config ai-core/configs/cutout_dataset_coco_voc.json
```

Output:

```text
ai-core/data/cutout_mix_512/
  images/
  annotations.json
  quality_report.json
```

Train with:

```bash
cd ai-core
./scripts/train.sh \
    --annotation_file data/cutout_mix_512/annotations.json \
    --image_dir data/cutout_mix_512/images \
    --output_dir ./weights/mask2former-cutout-coco-voc-v1
```

## Smoke Test

Use the legacy HVM source only for script validation, not for final training.
This smoke test requires `ai-core/data/hvm_coco_512/` to exist locally:

```bash
uv run --project ai-core python ai-core/scripts/data/build_cutout_dataset.py \
    --output-dir /tmp/cutout_dataset_smoke \
    --clean-output \
    --coco hvm ai-core/data/hvm_coco_512/annotations.json ai-core/data/hvm_coco_512/images \
    --max-total 12 \
    --max-both 0 \
    --max-person-only 6 \
    --max-car-only 6 \
    --max-negative 0 \
    --min-both-ratio 0 \
    --min-selected 12 \
    --resize-mode stretch \
    --blur-threshold 0
```
