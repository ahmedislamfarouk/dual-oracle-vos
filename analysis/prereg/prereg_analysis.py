"""Analysis of the pre-registered runs (prereg/PREREGISTRATION.md). Usage: python prereg_analysis.py OUTDIR...

Merges the two shards of every job, then runs the pre-registered paired tests. A video missing from
either arm is dropped from that comparison. Bootstrap: 20,000 resamples, seed 0.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

def load(dirs, job, arm):
    out = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, f"{job}_s*.json"))):
            pv = json.load(open(p)).get("per_video", {})
            out.update({k: v[arm]["jf"] for k, v in pv.items() if arm in v})
    return out

def pair(A, B):
    ids = sorted(set(A) & set(B))
    d = np.array([B[k] - A[k] for k in ids]) * 100
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(ids), a=float(np.mean([A[k] for k in ids]) * 100), b=float(np.mean([B[k] for k in ids]) * 100),
                d=float(d.mean()), ci95=np.percentile(m, [2.5, 97.5]).round(2).tolist(),
                ci90=np.percentile(m, [5, 95]).round(2).tolist(), p=float(p),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))

def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj.tolist()

def main(dirs):
    R = {}
    f0, q = load(dirs, "mose_qrd", "frame0"), load(dirs, "mose_qrd", "qrd")
    if q:
        o1, o4, o43 = load(dirs, "mose_o1", "o1"), load(dirs, "mose_o4b1", "o4"), load(dirs, "mose_o4b3", "o4")
        H = {"H1 QRD > frame0": pair(f0, q), "H2 O1 vs QRD (equiv 4)": pair(q, o1) if o1 else None,
             "H3 O4x1 vs QRD (equiv 4)": pair(q, o4) if o4 else None, "H4 O4x3 > QRD": pair(q, o43) if o43 else None}
        for k, v in H.items():
            if v and "equiv" in k:
                v["equivalent_4pp"] = bool(-4 < v["ci90"][0] and v["ci90"][1] < 4)
                # TOST p: max of two one-sided bootstrap tail probabilities is approximated by the CI rule above
        R["R1_mose_fresh"] = H
    lf, lq, l43 = load(dirs, "lvos_qrd", "frame0"), load(dirs, "lvos_qrd", "qrd"), load(dirs, "lvos_o4b3", "o4")
    if lq:
        R["R2_lvos_train"] = {"H6 QRD > frame0": pair(lf, lq), "H5 O4x3 > QRD": pair(lq, l43) if l43 else None}
    mf, mq, m4 = load(dirs, "mrcnn_qrd", "frame0"), load(dirs, "mrcnn_qrd", "qrd"), load(dirs, "mrcnn_o4b1", "o4")
    if mq:
        R["R3_maskrcnn"] = {"QRD > frame0": pair(mf, mq), "O4x1 vs QRD": pair(mq, m4) if m4 else None}
    df, dq = load(dirs, "davis_qrd", "frame0"), load(dirs, "davis_qrd", "qrd")
    if dq:
        R["R4_davis_check"] = {"means": {"frame0": np.mean(list(df.values())) * 100, "qrd": np.mean(list(dq.values())) * 100,
                                          "n": len(dq)}, "QRD > frame0": pair(df, dq)}
        secs = []
        for d in dirs:
            for p in glob.glob(os.path.join(d, "davis_qrd_s*.json")):
                pv = json.load(open(p))["per_video"]
                secs += [(v["seconds"], v["n_frames"], v["detector_calls"]) for v in pv.values()]
        if secs:
            s = np.array(secs, float)
            R["R4_davis_check"]["cost"] = {"sec_per_video": s[:, 0].mean(), "sec_per_frame": s[:, 0].sum() / s[:, 1].sum(),
                                           "detector_calls_per_video": s[:, 2].mean()}
    # Holm over the confirmatory hypotheses (one-sided superiority via the two-sided test, equivalence via its CI rule)
    conf = [(g, k) for g in ("R1_mose_fresh", "R2_lvos_train") if g in R for k, v in R[g].items() if v and "equiv" not in k]
    if conf:
        adj = holm([R[g][k]["p"] for g, k in conf])
        for (g, k), a in zip(conf, adj):
            R[g][k]["p_holm"] = a
    print(json.dumps(R, indent=1, default=float))

if __name__ == "__main__":
    main(sys.argv[1:])
