"""Core of the randomized approximate-sortedness algorithm.

beta-pairwise metric: fraction of index pairs (i < j) with L[i] <= L[j].
The estimator samples n pairs uniformly with replacement and reports the
fraction that are correctly ordered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    """A rotated list [split+1 .. N, 1 .. split] of N unique integers.

    Stored implicitly: elements are computed on demand, so N = 10**9 costs
    nothing. `beta` is known in closed form because the only out-of-order
    pairs are the split*(N-split) pairs straddling the rotation point.
    """

    size: int
    split: int

    @property
    def beta(self) -> float:
        n = self.size
        total = n * (n - 1) / 2
        incorrect = self.split * (n - self.split)
        return (total - incorrect) / total

    def at(self, idx: np.ndarray) -> np.ndarray:
        """Values at 0-based positions `idx` (vectorized)."""
        head = self.size - self.split  # length of the [split+1..N] chunk
        return np.where(idx < head, idx + self.split + 1, idx - head + 1)

    def to_array(self) -> np.ndarray:
        return self.at(np.arange(self.size, dtype=np.int64))


def problem_with_beta(size: int, beta: float) -> Problem:
    """Smallest split whose rotation gives sortedness closest to `beta`."""
    total = size * (size - 1) / 2
    target = (1.0 - beta) * total  # desired split*(size-split)
    disc = max(size * size - 4 * target, 0.0)
    split = int(round((size - np.sqrt(disc)) / 2))
    return Problem(size=size, split=min(max(split, 0), size))


def exact_beta(values: np.ndarray) -> float:
    """O(N^2)-equivalent ground truth via merge-sort inversion counting."""
    a = np.asarray(values)
    n = a.size
    total = n * (n - 1) // 2
    return (total - _inversions(a)) / total


def _inversions(a: np.ndarray) -> int:
    if a.size < 2:
        return 0
    mid = a.size // 2
    left, right = a[:mid], a[mid:]
    count = _inversions(left) + _inversions(right)
    ls, rs = np.sort(left), np.sort(right)
    # pairs (i in left, j in right) with left[i] > right[j]
    count += int(np.sum(ls.size - np.searchsorted(ls, rs, side="left")))
    return count


def approx_beta(problem: Problem, n: int, rng: np.random.Generator) -> float:
    """Estimate beta from n uniformly sampled pairs (with replacement)."""
    size = problem.size
    i = rng.integers(0, size, size=n)
    j = rng.integers(0, size, size=n)
    dup = i == j
    while dup.any():  # pairs must be of distinct positions
        j[dup] = rng.integers(0, size, size=int(dup.sum()))
        dup = i == j
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return float(np.count_nonzero(problem.at(lo) <= problem.at(hi)) / n)


def is_approximately_sorted(
    problem: Problem, alpha: float, n: int, rng: np.random.Generator
) -> bool:
    """YES iff the estimate reaches the requested threshold alpha."""
    return approx_beta(problem, n, rng) >= alpha


def chernoff_bound(n: int, t: float) -> float:
    """Pr[|beta' - beta| > t] <= 2 exp(-2 n t^2)."""
    return float(min(1.0, 2 * np.exp(-2 * n * t * t)))


def samples_for(t: float, delta: float) -> int:
    """n = ln(2/delta) / (2 t^2): samples needed for error t w.p. 1-delta."""
    return int(np.ceil(np.log(2 / delta) / (2 * t * t)))


def error_for(n: int, delta: float) -> float:
    """t = sqrt(ln(2/delta) / (2n)): error bound at n samples, w.p. 1-delta."""
    return float(np.sqrt(np.log(2 / delta) / (2 * n)))
