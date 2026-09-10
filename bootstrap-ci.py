import glob
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import cohen_kappa_score


def calculate_confidence_interval(data_series):
    data_array = np.asarray(data_series, dtype=float)

    res = bootstrap(
        (data_array,),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        random_state=42,
    )
    mean_value = data_array.mean()
    low_bound = res.confidence_interval.low
    high_bound = res.confidence_interval.high
    return mean_value, low_bound, high_bound

file_list = glob.glob("results/old/drift_*.csv")
# file_list = glob.glob("robustness/drift_*.csv")
#file_list = glob.glob("seedcheck/drift_*.csv")
sorted_files = sorted(file_list)

for file_path in sorted_files:
    df = pd.read_csv(file_path, encoding="utf-8")

    depth = int(df["pipeline_depth"][0])
    temp = float(df["temperature"][0])

    answerable_df = df[df["is_impossible"] == False]
    impossible_df = df[df["is_impossible"] == True]

    drift_mean, drift_lo, drift_hi = calculate_confidence_interval(df["global_drift_final"])
    #qa_mean, qa_lo, qa_hi = calculate_confidence_interval(answerable_df["final_qa_loss"])
    aqa_mean, aqa_lo, aqa_hi = calculate_confidence_interval(answerable_df["attributable_qa_loss"])
    tok_mean, tok_lo, tok_hi = calculate_confidence_interval(df["final_core_token_preservation"])
    fab_mean, fab_lo, fab_hi = calculate_confidence_interval(impossible_df["fabrication_rate"])
    unfaithful_condition = (answerable_df["rater_a_faithful"] == 0) | (answerable_df["rater_b_faithful"] == 0)
    faith_fail_rate = unfaithful_condition.mean()
    a = answerable_df["rater_a_faithful"]
    b = answerable_df["rater_b_faithful"]

    if a.nunique() > 1 or b.nunique() > 1:
        kappa = cohen_kappa_score(a, b) 
    else:
        kappa = float("nan")

    print(f"d{depth} T{temp} drift: {drift_mean:.3f}[{drift_lo:.3f},{drift_hi:.3f}]", end="  ")
    # print(f"d{depth} T{temp} qa: {qa_mean:.3f}[{qa_lo:.3f},{qa_hi:.3f}]", end="  ")
    print(f"d{depth} T{temp} aqa: {aqa_mean:.3f}[{aqa_lo:.3f},{aqa_hi:.3f}]", end="  ")
    print(f"d{depth} T{temp} fab: {fab_mean:.3f}[{fab_lo:.3f},{fab_hi:.3f}]", end="  ")
    print(f"d{depth} T{temp} tok: {tok_mean:.3f}[{tok_lo:.3f},{tok_hi:.3f}]", end="  ")
    print(f"faith_fail: {faith_fail_rate:.3f}")
    print(f"kappa: {kappa:.3f}")