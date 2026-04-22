# Training Notebook Guide

Use this file to choose between the two training notebooks in `notebooks/`.

## Use `notebooks/colab_train.ipynb` when

- you want the strongest available training hardware
- you want to use Google Colab GPU runtime
- you are okay downloading Food-101 into Colab
- you want the fastest path to a raw `resnet50` bundle on hosted hardware

This notebook is the recommended option if your main goal is:

- train `resnet50_raw`
- export `production_bundle`
- use it in the Django API with `SNAPCAL_ENABLE_SEGMENTATION=false`

## Use `notebooks/colab_train_local.ipynb` when

- you want to run everything on your own machine
- you already have the repo locally
- you may already have the Food-101 dataset locally
- you want a smaller local training run
- you are on macOS and want to use Apple MPS if available

This notebook creates a lighter local config by default:

- `resnet50_raw_local`
- fewer epochs
- smaller batch size
- no segmentation

It also assumes Food-101 already exists on your machine. Point `FOOD101_ROOT` at the dataset root or place the dataset under `data/raw` so the notebook can auto-detect it.

It is intended for local experimentation and generating a local `production_bundle` without Colab.

## Which one should you pick?

Choose `colab_train.ipynb` if:

- you want better performance than your laptop
- you are okay with a cloud notebook workflow

Choose `colab_train_local.ipynb` if:

- you want a self-contained local workflow
- you want to avoid Colab resets and Drive copying
- you are testing on your Mac first

## After either notebook finishes

Both notebooks are designed to produce a raw-model bundle for local Django usage.

After export:

1. Place or keep the bundle at `artifacts/models/production_bundle`
2. Run Django with `SNAPCAL_ENABLE_SEGMENTATION=false`
3. Follow `local-run.md` for backend and frontend startup

## Related files

- `notebooks/colab_train.ipynb`
- `notebooks/colab_train_local.ipynb`
- `local-run.md`
