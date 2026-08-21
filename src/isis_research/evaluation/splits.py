"""Reel- and station-disjoint evaluation, and uncertainty by deployment unit.

One film reel supplies up to 117 scans that share film stock, exposure and
scanner pass, so a scan-level split leaks badly: the model meets the test
reel's own siblings during training and the held-out score measures memory
rather than generalization. Every split here groups by reel, and
`check_disjoint` is the assertion that says so out loud.

The deployment unit is the reel and the station, not the scan. A median over
scans is dominated by whichever reel contributed most of them - Resolute
supplies 943 of the held-out scans and Kashima 17 - so a scan-level
confidence interval describes the biggest reel, not the archive. The
bootstrap here resamples the *groups*.

Seed variation is not dataset uncertainty. Re-running a fit with a different
random seed measures the optimizer, not the archive, and must never be
reported as a confidence interval.
"""

from __future__ import annotations

import numpy as np


def groups_of(records, key="reel"):
    return sorted({record[key] for record in records})


def check_disjoint(fit_records, test_records, key="reel"):
    """Raise if any group appears on both sides of a split."""
    leaked = sorted(
        set(groups_of(fit_records, key)) & set(groups_of(test_records, key))
    )
    if leaked:
        raise ValueError(
            f"train/test leakage: {len(leaked)} {key}(s) on both sides: {leaked[:5]}"
        )
    return True


def grouped_folds(records, folds, seed=0, key="reel"):
    """Yield (fit, test) folds that are disjoint by `key`.

    Groups are dealt largest-first into the currently smallest bucket, so the
    folds carry comparable scan counts even though reels differ in size by two
    orders of magnitude.
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
    """Yield (fit, test, group) holding out one whole group at a time.

    This is leave-one-reel-out and leave-one-station-out both, depending on
    `key`. Groups smaller than `min_test` are skipped rather than reported as
    a fold whose score is one scan wide.
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
    """-> (statistic, low, high) by resampling `values` with replacement.

    `low`/`high` are None below `min_n`, because a two-point interval invites
    a confidence claim the sample cannot support.
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
    """Per-group n and bootstrapped statistic, for the report table.

    The interval is over scans within the group; `overall` bootstraps the
    per-group statistics themselves, so one large reel cannot set the width.
    """
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
