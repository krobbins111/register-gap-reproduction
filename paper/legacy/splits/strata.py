"""Deterministic, image-disjoint stratified train/val/test splitter.

Several grid datasets (Fashion200k, SciCap, FACAD, Recipe1M, …) ship as one big
undivided pool. The projection experiment needs the SAME train/val/test contract
every other dataset obeys:

  * splits are **image-level disjoint** (the leakage audit checks exactly this),
  * sizes are controlled (the grid pipeline's target: 50k train / 5k val / 5k test, clamped
    to what the pool supports),
  * composition is **stratified** by a metadata key (scene class, product type,
    figure type, …) so train and test cover the same categories in the same
    proportions — otherwise a class that lands only in test inflates the gap.

`stratified_split` takes a list of records, a function that returns each
record's stratum key, target sizes, and a seed; it returns three disjoint index
lists. It is pure/inputs-only (no I/O), so it's unit-testable and every dataset
loader that calls it gets identical, reproducible behavior.

Disjointness note: records are the unit of splitting, and each record carries one
image, so record-disjoint == image-disjoint AS LONG AS a given image appears in
at most one record. Loaders must therefore dedupe images to one record before
calling this (e.g. one caption per image), which every loader here does.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

# the grid pipeline's target sizes (Jul 2026). Clamped per-pool: a small dataset just uses a
# proportional share (see _clamp_sizes), so SkinCAP (~4k) or RSICD-scale pools
# don't error — they run at natural size.
DEFAULT_SIZES = (50_000, 5_000, 5_000)   # (train, val, test)


def _clamp_sizes(n_total: int, sizes: tuple[int, int, int]
                 ) -> tuple[int, int, int]:
    """Fit (train, val, test) into n_total without overlap.

    If the pool is big enough, return the targets untouched. If not, keep the
    val/test targets (capped at ~15% of the pool each so they can't eat the
    whole set) and give train whatever remains. This keeps eval sets a sane,
    comparable size across datasets while letting train shrink gracefully.
    """
    tr, va, te = sizes
    if n_total >= tr + va + te:
        return tr, va, te
    va = min(va, max(1, int(0.15 * n_total)))
    te = min(te, max(1, int(0.15 * n_total)))
    tr = max(0, n_total - va - te)
    return tr, va, te


def stratified_split(
    records: Sequence[T],
    key_fn: Callable[[T], object],
    *,
    sizes: tuple[int, int, int] = DEFAULT_SIZES,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_idx, val_idx, test_idx) into `records`.

    Disjoint, deterministic (seeded), and stratified by `key_fn(record)`. Sizes
    are clamped to the pool via `_clamp_sizes`. Within each stratum, indices are
    shuffled once (seeded) and dealt test → val → train proportionally, so every
    stratum contributes to every split in proportion to its frequency.
    """
    n = len(records)
    tr, va, te = _clamp_sizes(n, sizes)
    want_total = tr + va + te

    # Bucket record indices by stratum.
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[key_fn(rec)].append(i)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    # Deal each stratum proportionally. Process strata in a stable, seeded order
    # so the result is reproducible regardless of dict iteration order.
    for key in sorted(buckets, key=lambda k: (str(type(k)), str(k))):
        idxs = buckets[key]
        rng.shuffle(idxs)                         # in place, seeded
        frac = len(idxs) / n
        # This stratum's quota for each split (fractional, floored; leftovers
        # go to train, the elastic split).
        s_te = int(round(te * frac))
        s_va = int(round(va * frac))
        s_te = min(s_te, len(idxs))
        s_va = min(s_va, len(idxs) - s_te)
        test_idx.extend(idxs[:s_te])
        val_idx.extend(idxs[s_te:s_te + s_va])
        # Train gets the rest of this stratum, but only up to the global train
        # quota's proportional share (so we don't overshoot want_total wildly).
        rest = idxs[s_te + s_va:]
        train_idx.extend(rest)

    # If clamping asked for fewer train than the leftover supplies, trim train
    # deterministically (shuffle-then-cut) so the requested size is honored.
    if len(train_idx) > tr:
        rng.shuffle(train_idx)
        train_idx = train_idx[:tr]

    # Final safety: guarantee disjointness (defensive; construction already is).
    assert not (set(train_idx) & set(val_idx)), "train/val overlap"
    assert not (set(train_idx) & set(test_idx)), "train/test overlap"
    assert not (set(val_idx) & set(test_idx)), "val/test overlap"

    train_idx.sort(); val_idx.sort(); test_idx.sort()
    print(f"[strata] pool={n} -> train {len(train_idx)} / val {len(val_idx)} / "
          f"test {len(test_idx)}  (targets {tr}/{va}/{te}, "
          f"{len(buckets)} strata, seed={seed})")
    return train_idx, val_idx, test_idx


def stratified_subsample(
    records: Sequence[T],
    key_fn: Callable[[T], object],
    k: int,
    *,
    seed: int = 0,
) -> list[int]:
    """Pick ~k record indices, stratified by key_fn (proportional per stratum).

    For datasets that already have NATIVE train/val/test splits but are far
    larger than the target size (SciCap ~400k): subsample each native split down
    to the target while preserving its category mix. Returns sorted indices. If
    k >= len(records), returns all indices.
    """
    n = len(records)
    if k >= n:
        return list(range(n))
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[key_fn(rec)].append(i)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in sorted(buckets, key=lambda x: (str(type(x)), str(x))):
        idxs = buckets[key]
        rng.shuffle(idxs)
        take = int(round(k * len(idxs) / n))
        chosen.extend(idxs[:min(take, len(idxs))])
    # Rounding can under/overshoot; correct to exactly k deterministically.
    if len(chosen) > k:
        rng.shuffle(chosen); chosen = chosen[:k]
    elif len(chosen) < k:
        remaining = sorted(set(range(n)) - set(chosen))
        rng.shuffle(remaining)
        chosen.extend(remaining[:k - len(chosen)])
    chosen.sort()
    return chosen


def assign_split(
    records: Sequence[T],
    key_fn: Callable[[T], object],
    rid_fn: Callable[[T], str],
    *,
    cache_path: Path,
    sizes: tuple[int, int, int] = DEFAULT_SIZES,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Compute (and cache) a stable train/val/test assignment of RECORD IDS.

    `rid_fn(record)` must return a STABLE, unique id per record (a filename,
    item id, uuid, …) — the split is stored/returned as those ids, so the three
    independent load_split_pairs('<ds>', split) calls (train, then val, then
    test) all read the identical partition and a re-run is instant.

    Returns {"train": [rid, …], "val": […], "test": […]}. Because the ids are
    dataset-native and image-unique, the partition is image-disjoint. The cache
    is keyed by (seed, sizes) via `cache_path`; delete it to reshuffle.
    """
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if (cached.get("seed") == seed and tuple(cached.get("sizes", ())) == tuple(sizes)
                and cached.get("n_pool") == len(records)):
            return {k: cached[k] for k in ("train", "val", "test")}
        print(f"[strata] cache {cache_path.name} stale (seed/sizes/pool changed) "
              f"— recomputing")

    tr_i, va_i, te_i = stratified_split(records, key_fn, sizes=sizes, seed=seed)
    assign = {
        "train": [rid_fn(records[i]) for i in tr_i],
        "val": [rid_fn(records[i]) for i in va_i],
        "test": [rid_fn(records[i]) for i in te_i],
    }
    # Sanity: ids must be unique and disjoint across splits (image-disjoint).
    allids = assign["train"] + assign["val"] + assign["test"]
    if len(set(allids)) != len(allids):
        raise ValueError("assign_split: rid_fn is not unique across records — "
                         "image-disjointness cannot be guaranteed.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "sizes": list(sizes), "n_pool": len(records),
                   **assign}, f)
    print(f"[strata] wrote split assignment -> {cache_path.name}")
    return assign
