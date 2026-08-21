from pathlib import Path

import numpy as np

from scripts.training.continual_train_phase6_models import (
    SampleRef,
    balanced_refs,
    split_pool,
    validation_better,
)


def _ref(index, station):
    return SampleRef(
        Path("corpus"),
        Path("targets"),
        {"pair_name": f"pair-{index}", "csa_artifact": "a.npz"},
        "target.npz",
        station,
        "1.0-9.0MHz",
        ">=0.99",
    )


def test_continual_split_is_repeatable_and_sampling_keeps_buckets_visible():
    pool = [_ref(index, "RES") for index in range(7)] + [_ref(7, "KSH")]
    train, validation = split_pool(pool, 0.25, 42)
    train_again, validation_again = split_pool(pool, 0.25, 42)

    assert [ref.row["pair_name"] for ref in train] == [
        ref.row["pair_name"] for ref in train_again
    ]
    assert [ref.row["pair_name"] for ref in validation] == [
        ref.row["pair_name"] for ref in validation_again
    ]
    assert {ref.row["pair_name"] for ref in train}.isdisjoint(
        ref.row["pair_name"] for ref in validation
    )
    sampled = balanced_refs(pool, np.random.default_rng(4))
    assert len(sampled) == len(pool)
    assert {ref.station for ref in sampled} == {"RES", "KSH"}
    assert validation_better({"macro_mae": 0.1, "macro_correlation": 0.5}, None)
    assert not validation_better(
        {"macro_mae": 0.2, "macro_correlation": 0.9},
        {"macro_mae": 0.1, "macro_correlation": 0.5},
    )


def test_continual_validation_split_keeps_reels_together():
    pool = []
    for reel in ("reel-a", "reel-b", "reel-c", "reel-d"):
        for index in range(3):
            ref = _ref(f"{reel}-{index}", "RES")
            ref.row["reel"] = reel
            pool.append(ref)

    train, validation = split_pool(pool, 0.25, 42)

    assert validation
    assert {ref.row["reel"] for ref in train}.isdisjoint(
        {ref.row["reel"] for ref in validation}
    )
