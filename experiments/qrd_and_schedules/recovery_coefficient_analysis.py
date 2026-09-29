#!/usr/bin/env python3
"""Discovery-deficit recovery coefficient.

Tests whether the J&F recovered by temporal re-detection scales with the
first-frame discovery deficit (1 - committed recall of the frame-0-only
baseline), and whether the coefficient is (a) stable across the four schedules
and (b) transferable to a population it was not fitted on.

Inputs are the frozen CPD-2 merged arms. No GT is used to compute the deficit
predictor at deployment time beyond the baseline recall itself; the analysis
here is descriptive and uses archived GT to establish the relation.
"""
import json, argparse
import numpy as np
from scipy.stats import spearmanr

ARMS = ["fixed", "random", "global", "residual"]


def load(path):
    d = json.load(open(path))
    deficit, gain = [], []
    for v in d["videos"]:
        bm = v["baseline_metrics"]
        deficit.append(1.0 - bm["committed_recall"])
        gain.append(v["metrics"]["strict_jf"] - bm["strict_jf"])
    return np.asarray(deficit), np.asarray(gain)


def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(c)


def paired_bootstrap(x, Y, reps, seed):
    """Resample videos, refit all arms on the same resample (arms are paired)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty((reps, Y.shape[1]))
    for k in range(reps):
        i = rng.integers(0, n, n)
        A = np.vstack([x[i], np.ones(n)]).T
        out[k] = np.linalg.lstsq(A, Y[i], rcond=None)[0][0]
    return out


def unconstrained(x, y, reps, seed):
    """Fit gain = alpha + beta*deficit with bootstrap CIs on both terms."""
    rng = np.random.default_rng(seed)
    A = np.vstack([x, np.ones_like(x)]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    n = len(x)
    B = np.empty((reps, 2))
    for k in range(reps):
        i = rng.integers(0, n, n)
        B[k] = np.linalg.lstsq(np.vstack([x[i], np.ones(n)]).T, y[i], rcond=None)[0]
    r2 = 1 - ((y - A @ [beta, alpha]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(beta=float(beta), alpha=float(alpha), r2=float(r2),
                beta_ci=[float(v) for v in np.quantile(B[:, 0], [.025, .975])],
                alpha_ci=[float(v) for v in np.quantile(B[:, 1], [.025, .975])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ytvos-dir", required=True)
    p.add_argument("--lvos-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cells, slopes = [], []
    for arm in ARMS:
        yd, yg = load(f"{a.ytvos_dir}/ytvos_{arm}_merged.json")
        ld, lg = load(f"{a.lvos_dir}/lvos_{arm}_merged.json")
        m, c = fit(yd, yg)
        slopes.append(m)
        ry, py = spearmanr(yd, yg)
        rl, pl = spearmanr(ld, lg)
        pred = float((m * ld + c).mean())
        cells.append(dict(
            arm=arm, slope_ytvos=m, intercept_ytvos=c,
            spearman_ytvos=float(ry), p_ytvos=float(py),
            spearman_lvos=float(rl), p_lvos=float(pl),
            lvos_predicted_mean_gain=pred,
            lvos_actual_mean_gain=float(lg.mean()),
            lvos_abs_error=float(abs(pred - lg.mean())),
            n_ytvos=int(len(yd)), n_lvos=int(len(ld)),
        ))

    # dose-response ladder on the fixed arm
    yd, yg = load(f"{a.ytvos_dir}/ytvos_fixed_merged.json")
    ld, lg = load(f"{a.lvos_dir}/lvos_fixed_merged.json")
    ladder = []
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        my = (yd >= lo) & (yd < hi)
        ml = (ld >= lo) & (ld < hi)
        ladder.append(dict(
            deficit_lo=lo, deficit_hi=hi,
            ytvos_n=int(my.sum()),
            ytvos_mean_gain=float(yg[my].mean()) if my.sum() else None,
            lvos_n=int(ml.sum()),
            lvos_mean_gain=float(lg[ml].mean()) if ml.sum() else None,
        ))

    # unconstrained fits + paired slope-heterogeneity test on YouTube-VOS
    Ys, xs = [], None
    for arm in ARMS:
        xa, ya = load(f"{a.ytvos_dir}/ytvos_{arm}_merged.json")
        xs = xa if xs is None else xs
        assert np.allclose(xa, xs), "deficit must be baseline-derived and arm-invariant"
        Ys.append(ya)
    Y = np.column_stack(Ys)
    unc = {arm: unconstrained(xs, Y[:, j], 10000, 20260827) for j, arm in enumerate(ARMS)}
    B = paired_bootstrap(xs, Y, 10000, 20260827)
    obs = np.linalg.lstsq(np.vstack([xs, np.ones_like(xs)]).T, Y, rcond=None)[0][0]
    pairwise = []
    for j in range(len(ARMS)):
        for k in range(j + 1, len(ARMS)):
            d = B[:, j] - B[:, k]
            lo, hi = np.quantile(d, [.025, .975])
            pairwise.append(dict(a=ARMS[j], b=ARMS[k], diff=float(obs[j] - obs[k]),
                                 ci=[float(lo), float(hi)], resolved=bool(lo > 0 or hi < 0)))

    out = dict(
        schema_version="recovery-coefficient-v2",
        unconstrained_fits=unc,
        slope_pairwise=pairwise,
        slope_range_observed=float(obs.max() - obs.min()),
        n_resolved_pairwise=int(sum(p["resolved"] for p in pairwise)),
        definition=("gain = strict J&F(3 later calls) - strict J&F(frame-0 only); "
                    "deficit = 1 - committed recall of the frame-0-only baseline"),
        slope_mean=float(np.mean(slopes)), slope_sd=float(np.std(slopes)),
        slope_min=float(min(slopes)), slope_max=float(max(slopes)),
        cells=cells, dose_response=ladder,
    )
    json.dump(out, open(a.output, "w"), indent=2, sort_keys=True)
    print(f"slope mean {out['slope_mean']:.4f} sd {out['slope_sd']:.4f}; wrote {a.output}")


if __name__ == "__main__":
    main()
