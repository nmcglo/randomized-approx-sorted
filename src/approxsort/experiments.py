"""Experiments reproducing the empirical section of the original report.

Run `approxsort all` (or individual sub-commands). Figures and CSV data land
in `results/`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .core import (  # noqa: E402
    Problem,
    approx_beta,
    chernoff_bound,
    error_for,
    samples_for,
)

RESULTS = Path("results")
LIST_SIZE = 10**9
NUM_PROBLEMS = 20
TRIALS = 100


def problems(size: int = LIST_SIZE, count: int = NUM_PROBLEMS) -> list[Problem]:
    """The 20 benchmark rotations: beta sweeps 0.95 down to 0.50."""
    step = 0.5 / count
    return [Problem(size, int(round((1 - k * step) * size))) for k in range(1, count + 1)]


def default_n(size: int) -> int:
    return int(round(np.sqrt(size)))


def _writer(name: str, header: list[str], rows) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / name).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote results/{name}")


def _savefig(name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()
    print(f"wrote results/{name}")


def _betas(problem: Problem, n: int, trials: int, rng) -> np.ndarray:
    return np.array([approx_beta(problem, n, rng) for _ in range(trials)])


def expectation(rng, trials: int = TRIALS) -> None:
    """E[beta'] == beta: table of actual vs mean/min/max estimate."""
    n = default_n(LIST_SIZE)
    rows = []
    print(f"{'prob':>4} {'beta':>8} {'mean':>8} {'min':>8} {'max':>8} {'|diff|':>8}")
    for idx, prob in enumerate(problems(), start=1):
        b = _betas(prob, n, trials, rng)
        row = (idx, prob.beta, b.mean(), b.min(), b.max(), abs(prob.beta - b.mean()))
        rows.append(row)
        print(f"{idx:>4} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f} {row[4]:8.5f} {row[5]:8.5f}")
    _writer("expectation.csv", ["problem", "beta", "mean_bp", "min_bp", "max_bp", "abs_diff"], rows)

    actual = np.array([r[1] for r in rows])
    mean = np.array([r[2] for r in rows])
    lo = np.array([r[3] for r in rows])
    hi = np.array([r[4] for r in rows])
    x = np.arange(1, len(rows) + 1)
    plt.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o", capsize=3, label=r"$\beta'$ (mean, range)")
    plt.plot(x, actual, "kx", label=r"actual $\beta$")
    plt.xlabel("problem")
    plt.ylabel(r"$\beta$")
    plt.title(rf"Estimate vs. truth ({trials} trials, $n=\sqrt{{N}}$={n})")
    plt.legend()
    _savefig("expectation.png")


def variance(rng, trials: int = TRIALS, steps: int = 50) -> None:
    """Var[beta'] ~ beta(1-beta)/n, measured against the 1/(4n) bound."""
    prob = problems()[-1]  # beta = 0.5
    max_n = default_n(LIST_SIZE)
    ns = np.linspace(max_n / steps, max_n, steps).round().astype(int)
    variances = np.array([_betas(prob, int(n), trials, rng).var(ddof=1) for n in ns])
    _writer("variance.csv", ["n", "variance", "bound"], zip(ns, variances, 1 / (4 * ns)))

    plt.plot(ns, variances, "o-", label="measured")
    plt.plot(ns, prob.beta * (1 - prob.beta) / ns, "--", label=r"$\beta(1-\beta)/n$")
    plt.xlabel("samples $n$")
    plt.ylabel(r"Var[$\beta'$]")
    plt.title(rf"Variance of the estimate ($\beta$={prob.beta:.2f}, {trials} trials per $n$)")
    plt.legend()
    _savefig("variance.png")


def failure(rng, trials: int = 1000, step: float = 1e-4) -> None:
    """Failure rate vs. how far the input alpha sits from the true beta."""
    prob = problems()[-1]
    n = default_n(LIST_SIZE)
    betas = _betas(prob, n, trials, rng)
    alphas = np.arange(prob.beta - 0.02, prob.beta + 0.02 + step, step)
    # correct answer: YES iff alpha <= beta
    fails = np.array(
        [
            np.mean(betas < a) if a <= prob.beta else np.mean(betas >= a)
            for a in alphas
        ]
    )
    diff = alphas - prob.beta
    _writer("failure_rate.csv", ["alpha_minus_beta", "failure_rate"], zip(diff, fails))

    plt.plot(diff, fails)
    plt.xlabel(r"$\alpha_{in} - \beta$")
    plt.ylabel("failure rate")
    plt.title(rf"Failure rate vs. deviation ($n$={n}, {trials} trials)")
    _savefig("failure_rate.png")


def errors_vs_samples(rng, trials: int = 300, steps: int = 20) -> None:
    """Same failure envelope, now as a function of the sample size."""
    prob = problems()[-1]
    max_n = default_n(LIST_SIZE)
    ns = np.linspace(max_n / steps, max_n, steps).round().astype(int)
    alphas = np.arange(prob.beta - 0.05, prob.beta + 0.05, 1e-3)
    grid = np.zeros((len(alphas), len(ns)))
    for col, n in enumerate(ns):
        betas = _betas(prob, int(n), trials, rng)
        for row, a in enumerate(alphas):
            grid[row, col] = np.mean(betas < a) if a <= prob.beta else np.mean(betas >= a)

    plt.pcolormesh(alphas - prob.beta, ns, grid.T, shading="gouraud")
    plt.xlabel(r"$\alpha_{in} - \beta$")
    plt.ylabel("samples $n$")
    plt.title("Failure rate vs. deviation and sample size")
    plt.colorbar(label="failure rate")
    _savefig("errors_vs_samples.png")


def threshold(rng, trials: int = TRIALS, alpha_in: float = 0.75) -> None:
    """Biasing failures: shifting the decision threshold trades FN against FP."""
    n = default_n(LIST_SIZE)
    probs = problems()
    betas = {p.split: _betas(p, n, trials, rng) for p in probs}
    thresholds = np.arange(0.5, 1.0 + 1e-9, 0.001)
    total = len(probs) * trials
    fn, fp = [], []
    for at in thresholds:
        f_neg = sum(np.count_nonzero(betas[p.split] < at) for p in probs if p.beta >= alpha_in)
        f_pos = sum(np.count_nonzero(betas[p.split] >= at) for p in probs if p.beta < alpha_in)
        fn.append(f_neg / total)
        fp.append(f_pos / total)
    _writer("threshold.csv", ["alpha_thresh", "false_neg_rate", "false_pos_rate"], zip(thresholds, fn, fp))

    plt.plot(thresholds - alpha_in, fn, "r", label="false negatives")
    plt.plot(thresholds - alpha_in, fp, "b", label="false positives")
    plt.xlabel(r"$\alpha_{thresh} - \alpha_{in}$")
    plt.ylabel("mistakes / total trials")
    plt.title(rf"Threshold bias ($\alpha_{{in}}$={alpha_in}, $n$={n})")
    plt.legend()
    _savefig("threshold.png")


def bounds(_rng) -> None:
    """Chernoff numbers quoted in the report."""
    print(f"Pr[|b'-b| > 0.02] <= {chernoff_bound(10_000, 0.02):.5f}  (n=10,000)")
    print(f"n for t=0.01, delta=0.01: {samples_for(0.01, 0.01):,}")
    print(f"t for n=10,000, delta=0.01: {error_for(10_000, 0.01):.4f}")


COMMANDS = {
    "expectation": expectation,
    "variance": variance,
    "failure": failure,
    "errors": errors_vs_samples,
    "threshold": threshold,
    "bounds": bounds,
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print(f"\n== {name} ==")
        COMMANDS[name](rng)


if __name__ == "__main__":
    main()
