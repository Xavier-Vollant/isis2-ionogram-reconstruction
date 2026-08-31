"""Reel- and station-disjoint splits with group-level uncertainty.

Scans from one reel share film and scanner conditions, so splitting individual
scans can leak information. These helpers keep groups together and bootstrap
those groups rather than individual scans.
"""

from __future__ import annotations

import numpy as np


def groups_of(records, key="reel"):
    """Return sorted unique values for a split key."""
    return sorted({record[key] for record in records})


def check_disjoint(fit_records, test_records, key="reel"):
    """Raise if a group appears in both the fit and test records."""
    leaked = sorted(
        set(groups_of(fit_records, key)) & set(groups_of(test_records, key))
    )
    if leaked:
        raise ValueError(
            f"train/test leakage: {len(leaked)} {key}(s) on both sides: {leaked[:5]}"
        )
    return True


def grouped_folds(records, folds, seed=0, key="reel"):
    """Yield fit/test folds that are disjoint by `key`.

    Groups are assigned largest-first to the fold with the fewest records.
    """
    groups = groups_of(records, key)
    if len(groups) < folds:
        raise ValueError(f"need at least {folds} {key}s, found {len(groups)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    sizes = {group: sum(record[key] == group for record in records) for group in groups}
    buckets = [[] for _ in range(folds)]
    counts = [0] * folds
    for group in sorted(groups, key=lambda item: sizes[item], reverse=True):
        index = min(range(folds), key=counts.__getitem__)
        buckets[index].append(group)
        counts[index] += sizes[group]
    for bucket in buckets:
        held = set(bucket)
        fit = [record for record in records if record[key] not in held]
        test = [record for record in records if record[key] in held]
        check_disjoint(fit, test, key)
        yield fit, test


def leave_one_out(records, key="reel", min_test=1):
    """Yield fit/test records while holding out one group at a time.

    Groups smaller than `min_test` are skipped.
    """
    for group in groups_of(records, key):
        test = [record for record in records if record[key] == group]
        if len(test) < min_test:
            continue
        fit = [record for record in records if record[key] != group]
        if not fit:
            continue
        check_disjoint(fit, test, key)
        yield fit, test, group


def bootstrap_ci(
    values, statistic=np.median, resamples=2000, seed=0, alpha=0.05, min_n=3
):
    """Return a statistic and bootstrap interval for `values`.

    The interval is omitted when there are fewer than `min_n` values.
    """
    values = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=float
    )
    if len(values) == 0:
        return None, None, None
    point = float(statistic(values))
    if len(values) < min_n:
        return point, None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    spread = np.sort(statistic(values[draws], axis=1))
    low = float(spread[int(alpha / 2 * resamples)])
    high = float(spread[int((1 - alpha / 2) * resamples) - 1])
    return point, low, high


def group_summary(records, values, key="reel", statistic=np.median, seed=0):
    """Return per-group statistics and an overall group-level summary."""
    values = np.asarray(values, dtype=float)
    rows = []
    for group in groups_of(records, key):
        picked = [
            value
            for record, value in zip(records, values)
            if record[key] == group and np.isfinite(value)
        ]
        point, low, high = bootstrap_ci(picked, statistic, seed=seed)
        rows.append(
            {
                key: group,
                "n": len(picked),
                "value": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    per_group = [row["value"] for row in rows if row["value"] is not None]
    point, low, high = bootstrap_ci(per_group, statistic, seed=seed)
    return {
        "by_" + key: rows,
        "groups": len(rows),
        "overall": {"value": point, "ci_low": low, "ci_high": high, "over": f"{key}s"},
    }
