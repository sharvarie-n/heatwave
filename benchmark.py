"""
benchmark.py
Measures the claim in the IA-2 report: a B+ Tree range query is O(log n + k), while a scan
of an unordered list is O(n). Same data, same windows, both methods must return the same count.
"""
import random
import time

import pandas as pd

from bplus_index import BPlusTree

DEFAULT_SIZES = (10_000, 50_000, 100_000, 250_000)


def run_benchmark(sizes=DEFAULT_SIZES, window=30, tree_reps=300, scan_reps=15, seed=1):
    """One row per data size: average time of one range query (ms) for each method."""
    rows = []
    for n in sizes:
        rng = random.Random(seed)
        days = list(range(n))                      # one reading per day, oldest first
        t0 = time.perf_counter()
        tree = BPlusTree()
        for d in days:
            tree.insert((d, "x"), d)
        build_s = time.perf_counter() - t0
        flat = [(d, "x") for d in days]            # plain list, the "simple storage" baseline
        rng.shuffle(flat)                          # unordered, as a file/list of arrivals would be

        starts = [rng.randrange(0, n - window) for _ in range(max(tree_reps, scan_reps))]
        t0 = time.perf_counter()
        for a in starts[:tree_reps]:
            found, _ = tree.range_query(a, a + window - 1)
        tree_ms = (time.perf_counter() - t0) * 1000 / tree_reps

        t0 = time.perf_counter()
        for a in starts[:scan_reps]:
            hits = [k for k in flat if a <= k[0] <= a + window - 1]
        scan_ms = (time.perf_counter() - t0) * 1000 / scan_reps

        a = starts[0]                              # both methods must agree
        same = len(tree.range_query(a, a + window - 1)[0]) == sum(1 for k in flat if a <= k[0] <= a + window - 1)
        rows.append({
            "Records (n)": n, "Tree height": tree.height(),
            "B+ Tree query (ms)": round(tree_ms, 4), "List scan (ms)": round(scan_ms, 3),
            "Speed-up (x)": round(scan_ms / tree_ms, 1) if tree_ms else float("inf"),
            "Build time (s)": round(build_s, 2), "Same result": same,
        })
    return pd.DataFrame(rows)
