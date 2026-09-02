# AST-Boost

AST-Boost implements the v0.3 research specification for sign- and
basis-invariant spectral positional/structural encodings in Graph Transformers.
It is a GraphGPS plugin project: the intended contribution is the encoding, not
a replacement Transformer backbone.

The three supported variants are:

| Variant | Absolute encoding | Attention bias | Second-order fields |
| --- | --- | --- | --- |
| `lite` | MLP over sign-invariant local spectral features | invariant filtered spectral kernel | no |
| `kern` | first-order SignNet | invariant filtered spectral kernel | no |
| `full` | first-order SignNet | invariant filtered spectral kernel | low-frequency pair fields |

This remains a research prototype. It contains no claimed benchmark result;
all comparisons should use a matched GraphGPS budget and the protocol in
[`../READMEv0.3.md`](../READMEv0.3.md).

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

All three use the ZINC subset, `k=8`, and the same GraphGPS-sized backbone.
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

from ast_boost import ASTBoostPE, prepare_spectrum_batch

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

For a cache too large for VRAM, call `prepare_spectrum_batch(...,
pin_memory=True)` on CPU and move the resulting object with
`.to("cuda", non_blocking=True)`. The implementation precomputes invariant
first-order fields and legal second-order fields, evaluates all graphs' Bernstein
responses and spectral kernels together, and processes both SignNet signs in one
batched encoder call. Leave `contiguous=False` for interleaved/custom node layouts;
that compatibility path performs a stable node sort.

Prefer `forward_padded_batch` when dense attention memory is the constraint. It
allocates `(B, H, max_N, max_N)` bias rather than `(H, sum_N, sum_N)`.
`forward_batch` remains as the low-memory-independent compatibility/reference
path. In both cases, apply the returned mask before softmax. Compare the two
paths on the target workload with:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_gpu.py `
  --batch-size 32 --nodes 64 --variant full
```

The accepted/rejected experiments and the source recovery point are recorded in
[`docs/GPU_OPTIMIZATION.md`](docs/GPU_OPTIMIZATION.md).

For framework-independent offline caching, the installed package also provides
an NPZ-to-NPZ command.  The input archive contains `edge_index` plus optional
`edge_weight` and `num_nodes`; its output stores `eigenvalues`, `eigenvectors`,
and the v0.3 degeneracy blocks:

```bash
ast-boost-precompute graph.npz spectrum.npz --k 8 --laplacian sym
```

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
