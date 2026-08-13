# Experiment results

Every active training run is written to exactly one directory:

```text
results/<run-name>/
```

The path is anchored to the repository root, so it does not change with the
shell's current working directory. `train_model.py` accepts a single
`--run-name`; arbitrary output paths are intentionally unsupported.

Generated run contents and ZIP archives are ignored by Git. This README is the
only tracked file in this directory.

`package_results.py` creates a review ZIP only after held-out folds 1 and 2 are
both complete; one-fold smoke runs remain in their run directory but are not
published as complete OOF evidence.
