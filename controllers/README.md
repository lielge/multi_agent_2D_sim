# Saved controllers

Training publishes immutable inference-ready neural controllers here. Each
controller consists of a `.pt` model and a matching `.controller.json`
manifest. Use `examples/train_mlp.py controllers list` to inspect the catalog.

Do not rename one file without updating its manifest. Resumable optimizer
checkpoints belong in `artifacts/`, not this directory.
