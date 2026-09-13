"""Explicit graph batching policies for controlled training comparisons."""

import numpy as np


def batches(indices, counts, batch_size, *, seed=None, sampler="random"):
    """Visit each graph once; only training sortish batches correlate size.

    Evaluation sorts for padding efficiency, since eval-mode BatchNorm uses
    frozen running statistics. Random training is the GraphGym-style default.
    Sortish is an explicit ablation: reshuffling does not remove its size bias.
    """
    if batch_size < 1 or sampler not in {"random", "sortish"}:
        raise ValueError("positive batch_size and random/sortish sampler required")
    values = [int(index) for index in indices]
    rng = None if seed is None else np.random.default_rng(seed)
    if rng is None:
        buckets = [sorted(values, key=lambda index: counts[index])]
    else:
        shuffled = rng.permutation(values).tolist()
        if sampler == "random":
            buckets = [shuffled]
        else:
            window = 8 * batch_size
            buckets = [
                sorted(shuffled[offset : offset + window], key=lambda index: counts[index])
                for offset in range(0, len(shuffled), window)
            ]
    result = [
        bucket[offset : offset + batch_size]
        for bucket in buckets
        for offset in range(0, len(bucket), batch_size)
    ]
    if rng is not None and sampler == "sortish":
        rng.shuffle(result)
    return result
