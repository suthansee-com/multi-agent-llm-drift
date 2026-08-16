import glob, pandas as pd, numpy as np
from scipy.stats import bootstrap

def ci(v):
    v = np.asarray(v, float)
    if len(v) == 0 or np.all(v == v[0]): return v.mean() if len(v) else np.nan, np.nan, np.nan
    r = bootstrap((v,), np.mean, confidence_level=0.95, n_resamples=10000, random_state=42).confidence_interval
    return v.mean(), r.low, r.high

for f in sorted(glob.glob("results/drift_*.csv")):
    df = pd.read_csv(f, encoding="utf-8")
    d, t = int(df.pipeline_depth[0]), float(df.temperature[0])
    ans = df[df.is_impossible == False]; imp = df[df.is_impossible == True]
    for name, series in [("drift", ans.global_drift_final), ("qa", ans.final_qa_loss),
                         ("fab", imp.fabrication_rate), ("tok", ans.final_core_token_preservation)]:
        m, lo, hi = ci(series)
        print(f"d{d} T{t} {name}: {m:.3f}[{lo:.3f},{hi:.3f}]", end="  ")
    print(f"faith_fail: {((ans.rater_a_faithful==0)|(ans.rater_b_faithful==0)).mean():.3f}")