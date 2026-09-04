# AST-Boost

AST-Boost implements the core of the v0.3 research specification for sign- and
basis-invariant spectral positional/structural encodings in Graph Transformers.
It is a GraphGPS plugin project: the intended contribution is the encoding, not
a replacement Transformer backbone.

The three supported variants are:

| Variant | Absolute encoding | Attention bias | Second-order fields |
| --- | --- | --- | --- |
| `lite` | MLP over sign-invariant local spectral features | invariant filtered spectral kernel | no |
| `kern` | first-order SignNet | invariant filtered spectral kernel | no |
| `full` | first-order SignNet | invariant filtered spectral kernel | low-frequency pair fields |

This remains a research prototype, not an official GraphGPS reproduction.
A standalone GPS-style ZINC experiment runner is now available; its short pilot
results must not be presented as converged benchmark performance. All accuracy comparisons should
use a matched GraphGPS budget and the protocol in
[`../READMEv0.3.md`](../READMEv0.3.md). Read the
[theory-to-implementation audit](docs/THEORY_AUDIT.md) for corrected mathematical
claims, implementation boundaries, and measured synthetic GPU timings.

## Requirements and installation

Python 3.10 or newer is required. The spectral core depends only on PyTorch,
NumPy, SciPy, and pytest. Keep the test environment inside the project:

```powershell
cd main
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On Linux/macOS, replace the interpreter path with `.venv/bin/python`.

For PyG data handling, install the matching PyG wheels for the local
Torch/CUDA build and then request the extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pyg]"
```

For GraphGPS/GraphGym experiments:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[graphgps]"
```

The extras are deliberately separate: unit tests for the spectral modules do
not require a GPU, PyG, or a GraphGPS checkout.

### Project-local CUDA runtime

This checkout is verified on an NVIDIA GeForce RTX 5070 Laptop GPU with the
official PyTorch 2.13 CUDA 13.0 wheel. The CUDA build is installed only in
`main/.venv`; it does not modify the system Python. To recreate it after the
editable/development install above:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

Use the [PyTorch installation selector](https://docs.pytorch.org/get-started/locally/)
instead when the host requires a different CUDA build.

## Layout

```text
main/
├── benchmarks/           # CUDA timing and peak-memory comparison
├── configs/zinc/         # standalone configurations for the v0.3 variants
├── docs/                  # optimization decisions and recovery notes
├── src/ast_boost/        # spectral core and GraphGPS adapter
└── tests/                # invariance, degeneracy, and permutation tests
```

The configuration schema keeps the important v0.3 safety choices explicit:

- `precompute.laplacian: sym` uses the normalized symmetric Laplacian.
- `precompute.skip_zero: true` removes the zero eigenspace from LapPE.
- `k` is a minimum target: if its boundary falls inside a near-degenerate
  block, the cache retains the entire block rather than storing a basis-dependent
  partial subspace.
- `degeneracy.eps` is a relative gap threshold, not an absolute threshold.
- `kernel.block_clamp: true` evaluates the spectral response at a block mean,
  making the kernel invariant under an orthogonal basis rotation in a repeated
  or near-repeated eigenspace.
- `bias.standardize: true` standardizes each graph's off-diagonal kernel before
  applying a learnable attention scale.
- `fields.second_order` is only enabled by `ast_full.yaml`; it excludes
  every pair touching a non-singleton block (not just within-block pairs), and
  shares its GNN encoder across fields.  The first-order path represents such a
  block with `diag(P_B)` rather than arbitrary eigenvector columns.

## ZINC starting points

Use the configuration that matches the desired ablation:

```text
configs/zinc/ast_lite.yaml
configs/zinc/ast_kern.yaml
configs/zinc/ast_full.yaml
```

These configuration drafts target the ZINC subset, `k=8`, and the same
GraphGPS-sized backbone; they are not yet wired into a GraphGym config loader.
`ast_full.yaml` reduces the shared SignNet width so its additional second-order
path can be compared with `kern` under a similar parameter budget. Dataset
files and precomputed spectra are intentionally kept outside version control.

The public API follows the v0.3 interface sketch:

```python
from ast_boost import ASTBoostPE, precompute_spectrum

spectral = precompute_spectrum(data.edge_index, n=data.num_nodes, k=8)
# Run the dataset's categorical/raw node features through the usual GraphGPS
# node encoder first: ASTBoostPE operates on floating token embeddings.
node_tokens = node_encoder(data.x).float()  # e.g. width 64
pe = ASTBoostPE(
    heads=4,
    pe_dim=32,
    k0_pairs=4,
    token_dim=64,       # project concatenated PE back to GPS width
    variant="full",
)
x_pe, attn_bias = pe(node_tokens, spectral, data.edge_index)
```

`U` and eigenvalues are precomputed and frozen. Training should optimize only
the kernel response, SignNet encoders, adapter, and GraphGPS backbone.
For a disjoint PyG batch, use `forward_batch`; it returns
`(tokens, bias, same_graph_mask)`.  Pass that mask to `add_attention_bias` (or
the equivalent GraphGPS attention hook) so cross-graph logits are set to
`-inf`; a zero cross-graph bias is not itself a mask.
The `spectra` sequence passed to `forward_batch` follows ascending graph IDs in
the PyG `batch` vector (the standard `Batch.from_data_list` order).

### GPU-optimized training path

Pack each minibatch into one `TorchSpectrumBatch` rather than launching the
spectral modules once per graph. If a batch is reused, cache this object; for a
DataLoader, create it in the collate path and transfer the whole object once:

```python
import torch

from ast_boost import ASTBoostPE, attention_softmax, prepare_spectrum_batch

device = torch.device("cuda")
spectrum_batch = prepare_spectrum_batch(batch_spectra, k0_pairs=4, device=device)
pe = ASTBoostPE(
    variant="full", heads=4, pe_dim=32, k0_pairs=4, token_dim=64
).to(device)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    tokens, bias, mask, valid_nodes = pe.forward_padded_batch(
        node_tokens.to(device, non_blocking=True),
        edge_index.to(device, non_blocking=True),
        spectrum_batch,
        batch.to(device, non_blocking=True),
        contiguous=True,  # standard PyG Batch.from_data_list layout
    )
```

`tokens` stays flat in the original node order. Pack the backbone's Q/K/V into
`(B, H, max_N, head_dim)` using `valid_nodes` before using the padded bias.
For attention logits of shape `(B, H, max_N, max_N)`, use:

```python
weights = attention_softmax(logits, bias, attention_mask=mask)
```

This returns zero probability for fully masked padding-query rows, with finite
gradients. Applying ordinary softmax to an all-`-inf` row would produce NaNs.
Frozen spectra and kernel contractions stay in at least float32 under AMP;
neural PE layers still benefit from mixed precision.

For a cache too large for VRAM, call `prepare_spectrum_batch(...,
pin_memory=True)` on CPU and move the resulting object with
`.to("cuda", non_blocking=True)`. The implementation precomputes invariant
first-order fields and legal second-order fields, evaluates all graphs' Bernstein
responses and spectral kernels together, and processes both SignNet signs in one
batched encoder call. Leave `contiguous=False` for interleaved/custom node layouts;
that compatibility path performs a stable node sort.

Prefer `forward_padded_batch` when dense attention memory is the constraint. It
allocates `(B, H, max_N, max_N)` bias rather than `(H, sum_N, sum_N)`.
`forward_batch` remains as the compatibility/reference path and allocates a
larger disjoint bias. In both cases, use the returned mask with attention.
Compare the two
paths on the target workload with:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py `
  --batch-size 32 --nodes 64 --variant full --repeats 5
```

Add `--include-preparation` to include per-step packing of GPU-resident spectra,
or `--backward` to measure forward plus backward (without optimizer/GradScaler).
Results report medians and ranges with alternating path order. They do not
include the GraphGPS backbone, DataLoader, or disk IO.

The accepted/rejected experiments and the source recovery point are recorded in
[`docs/GPU_OPTIMIZATION.md`](docs/GPU_OPTIMIZATION.md) and the latest
[`docs/THEORY_AUDIT.md`](docs/THEORY_AUDIT.md).

For framework-independent offline caching, the installed package also provides
an NPZ-to-NPZ command.  The input archive contains `edge_index` plus optional
`edge_weight` and `num_nodes`; its output stores `eigenvalues`, `eigenvectors`,
and the v0.3 degeneracy blocks:

```bash
ast-boost-precompute graph.npz spectrum.npz --k 8 --laplacian sym
```

The lowest-spectrum solver now uses a negative shift (`--sparse-sigma=-1e-5`).
Dense and sparse branches both coalesce duplicate directed edges by maximum
weight before symmetrizing. Regenerate caches affected by the former positive
shift or inconsistent duplicate-edge handling; the sample YAMLs use a new
`spectral_k8_v2` directory without deleting old caches.

For the combinatorial Laplacian ablation, set `ASTBoostPE(kernel_domain_max=...)`
to a suitable shared upper spectral bound; its eigenvalues need not lie in
`[0, 2]`. `kernel_eps` controls the off-diagonal standardization floor (default
`1e-8`); near-constant kernels still require sensitivity checks.

## Controlled ZINC experiments

Full is retained as a first-class variant. Short runs, a single seed, or extra
runtime are not grounds to delete it. The runner has no automatic elimination
of variants or checkpoints.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[experiments]"
.\.venv\Scripts\python.exe -m ast_boost.experiments.train `
  --output runs/my-zinc-pilot --epochs 5 --seeds 42 43 44 45 46
```

This runs `rwse`, random-sign `lappe`, `signnet_local`, `kern`, and `full` on
the same 10-layer GINE+Transformer-style backbone and the official ZINC subset
splits. `signnet_local` is the project's first-order, degeneracy-aware control,
NOT the public SignNet implementation. The old YAML drafts are not consumed by
this CLI; its actual arguments and source hashes are saved in each run manifest.
See [experiment protocol and limits](docs/EXPERIMENTS.md).
The completed 25-run pilot and layout-reuse timings are reported in
[ZINC pilot results](docs/ZINC_PILOT_20260903.md); Full remains available.

Datasets, GPU-bank cache, metrics and checkpoints remain under `main/data`,
`main/.cache`, and your `main/runs` output. Resume an interrupted run using the
identical command plus `--resume`; changed code/protocol or an existing output
without `--resume` is rejected. Test-set evaluation is opt-in via
`--evaluate-test`; leave it off while choosing hyperparameters.

The GPU-bank path uses `ASTBoostPE.forward_packed` to reuse the node/edge layout.
It preserves `forward_padded_batch`'s values and gradients and keeps both of
Full's field branches. The caller must supply already-validated padded edges.

The local trainer now defaults to `--node-layout compact`: GINE, BatchNorm and
feed-forward layers work on real nodes, while attention alone packs Q/K/V.
`--node-layout padded` retains the previous backbone implementation for
comparison. Parameters and checkpoint tensor names are unchanged; floating-point
rounding and dropout RNG consumption can differ. Use a **new output directory**
for new-code experiments, not the completed pilot directory.

`--k 8 --pairs 4 --rw-steps 20` exposes the spectral/RWSE ablation budgets without
editing code; `pairs` is the low-frequency cutoff k0, not the number of pair
fields. Defaults are unchanged. Each new study automatically saves and verifies
`source_snapshot.zip`. Epoch commits now include metric history and best-model
weights, so interrupted log/best-checkpoint publication can be repaired on resume.
See [compact-backbone optimization and verification](docs/COMPACT_BACKBONE_20260903.md).

For joint accuracy/throughput experiments, `--field-scaling size` conditions the
first-order inputs by sqrt(N) and second-order inputs by N without changing the
relative kernel or removing any Full fields. It is opt-in; the default remains
`none`. `--signal-backend dense` is an optional small-graph GEMM implementation
with extra adjacency storage; `sparse` remains the default. Neither flag alone
constitutes evidence of better accuracy or runtime.

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.compare `
  --output runs/my-accuracy-speed-study --epochs 20 --seeds 42 43 44
```

This freezes three Full configurations before training, rotates their order by
seed, and preserves every result. It separates a field-conditioning ablation
from a larger-batch/learning-rate configuration. See the
[accuracy and throughput study](docs/ACCURACY_SPEED_20260904.md) for its evidence
and limits; test labels are not used for this optimization.

The larger-batch arm with `lr=0.002` is exploratory: an independent 30-epoch
run encountered exploding gradients and remains recorded as incomplete. Do not
disable nonfinite-gradient checks to force it through. The report also records
the separate B64/`lr=0.001` follow-up and both beneficial and harmful BN
recalibration results; no original checkpoint or Full branch is replaced.

The latest local ZINC candidate combines exact shared-field execution, B128,
and a 35-epoch cosine schedule:

```powershell
.\.venv\Scripts\python.exe -m ast_boost.experiments.train `
  --methods full --field-scaling size --batch-size 128 --lr 0.001 `
  --scheduler cosine --epochs 35 --seeds 42 43 44 `
  --output runs/my-full-accuracy-speed
```

Across three seeds, it obtained validation MAE 0.2827 versus the previous B64/20
candidate's 0.3000, while mean training+validation time fell from 103.6 to 97.8
seconds on this device. It used 35 data passes but 11.9% fewer optimizer updates;
peak allocated memory increased to about 483 MiB. The earlier same-source,
same-20-epoch schedule comparison remains the evidence that favored cosine over
Plateau. These are exploratory validation results, not test-set or published
benchmark claims. `plateau` remains the compatibility default; request `cosine`
explicitly. All Full branches and parameters remain active. See the
[latest optimization report](docs/NEXT_OPTIMIZATION_20260904.md) and the earlier
[mathematical/code optimization report](docs/MATH_CODE_OPTIMIZATION_20260904.md).

Full now executes first- and second-order fields in one call when their `psi`
GIN is shared, as required by the v0.3 batching design. The two independent
`rho` readouts and every field remain. Set `fuse_shared_fields=False` only for
the retained reference path. Float64 values and gradients match; repeated
whole-model BF16 measurements reduced training-step latency by 4.66%–5.33%.

## Verification

Run the complete test suite after installation:

```powershell
.\.venv\Scripts\python.exe -m pytest -W error
```

The required v0.3 invariance checks can also be run individually:

```powershell
.\.venv\Scripts\python.exe -m pytest -m sign_invariance
.\.venv\Scripts\python.exe -m pytest -m degeneracy
.\.venv\Scripts\python.exe -m pytest -m permutation
.\.venv\Scripts\python.exe -m pytest -m kernel_invariance
```

Do not start ZINC training until these tests pass. In particular,
`kernel_invariance` protects against the subtle error of applying different
filter values to vectors inside a degenerate spectral block.
