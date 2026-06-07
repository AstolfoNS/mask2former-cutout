# AI Core Scripts

This directory keeps operational scripts for the AI core project.

```text
scripts/
    data/
        build_cutout_dataset.py
        download_and_build_coco_cutout.py
        prepare_pascal_voc2012.py
    setup_env.sh
    train.sh
```

- `data/`: dataset download, conversion, filtering, and build scripts.
- `data/prepare_pascal_voc2012.py`: downloads and prepares Pascal VOC 2012
  semantic segmentation data for the COCO+VOC training mix.
- `setup_env.sh`: local AI core environment setup.
- `train.sh`: training entrypoint wrapper.

Run commands from the repository root unless a script explicitly says
otherwise.

Default Python entrypoints should use uv instead of directly calling a virtual
environment interpreter:

```bash
uv run --project ai-core python ai-core/scripts/data/build_cutout_dataset.py --help
cd ai-core
./scripts/train.sh --help
```
