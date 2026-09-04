# GPU optimization notes

This page records the 2026-09-02 pass. The follow-up
[theory audit](THEORY_AUDIT.md) records the 2026-09-03 backup, numerical fixes,
CUDA device-alias fix, and repeated inference/backward measurements.
The later [compact-backbone report](COMPACT_BACKBONE_20260903.md) covers
whole-model training/inference timing and trained-checkpoint equivalence;
its timings must not be mixed with the encoder-only measurements below.

## Recovery point

The source snapshot taken before this optimization pass is:

```text
../backups/main-source-pre-opt-20260902.zip
```

It excludes `.venv`, datasets, and generated caches. Extract it to a separate
directory for inspection rather than overwriting the working tree directly:

```powershell
Expand-Archive ..\backups\main-source-pre-opt-20260902.zip `
  -DestinationPath ..\restore-preview
```

## Accepted changes

- `TorchSpectrumBatch` pads eigenpairs and invariant fields once and moves them
  between CPU/CUDA as one reusable object.
- `forward_padded_batch` executes SignNet and spectral kernels across the whole
  minibatch rather than once per graph.
- `contiguous=True` skips sorting for the standard PyG
  `Batch.from_data_list` layout. The default path still supports interleaved
  graph nodes with a stable sort.
- CUDA AMP output allocation follows the computed dtype, while bias statistics
  remain in float32 for numerical stability.
- The legacy `forward_batch` path remains available as a correctness reference
  and rollback option.

## Rejected experiment

Fusing first- and second-order AST-Full fields into one shared GIN call was
tested and reverted. It did not improve step time on the target GPU and raised
the measured peak allocation from about 59 MiB to 79 MiB because both field
families' hidden states remained live together.

## Reproduce validation

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache src tests benchmarks
.\.venv\Scripts\python.exe -m pytest -W error -p no:cacheprovider
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py `
  --batch-size 32 --nodes 64 --iterations 50 --variant full
```

Run performance measurements with other GPU-heavy applications closed. The
benchmark reports spectrum-batch construction separately from steady-state step
time and compares both paths in the same process.
