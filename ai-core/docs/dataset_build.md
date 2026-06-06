# Cutout Dataset Build Guide

This project should train on compact, real-scene person/car data rather than
the old single-instance HVM-only dataset.

## Target Mix

The default recipe builds about 1000 images:

- 550 images containing both person and car.
- 175 person-only images.
- 175 car-only images.
- Up to 100 negative/background images if provided.

The builder enforces a minimum selected count and a minimum person+car ratio so
an HVM-only dataset cannot accidentally pass as production-quality training
data.

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
```

COCO should be the first source to add because it provides real person/car
instance masks with the lowest conversion cost. Cityscapes is useful for street
scene hard cases.

## Build

One-command COCO small subset download and build:

```bash
uv run --project ai-core python ai-core/scripts/data/download_and_build_coco_cutout.py \
    --clean-output
```

This command downloads only the COCO annotations plus selected person/car
images. It does not download the full train2017 image archive.

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

Manual build from already downloaded sources:

```bash
uv run --project ai-core python ai-core/scripts/data/build_cutout_dataset.py \
    --config ai-core/configs/cutout_dataset_small.json
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
uv run python -m src.train \
    --annotation_file data/cutout_mix_512/annotations.json \
    --image_dir data/cutout_mix_512/images
```

## Smoke Test

Use the existing HVM source only for script validation, not for final training:

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
