import random
import streamlit as st
import json
import pandas as pd
import numpy as np
import re
import operator
import math
import ast
from sklearn.metrics import cohen_kappa_score
from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import datetime
import plotly.graph_objects as go

from pipeline import (
    AgentState,
    calculate_core_token_preservation,
    extract_content_tokens,
    create_pipeline,
    init_nltk,
    lemmatizer,
    rate_faithfulness_A,
    rate_faithfulness_B,
    calculate_semantic_drift,
    extract_agent_understanding,
    SEMANTIC_SHIFT_THRESHOLD,
    AGENT_MODEL,
    READER_MODEL,
    RATER_MODEL_A,
    RATER_MODEL_B,
    EMBEDDER_MODEL,
    rate_faithfulness_at_checkpoints,
    select_checkpoints,
    measure_f1_recoverability_loss,
    measure_em_recoverability_loss,
    measure_qa_recall_loss,
)


@st.cache_resource
def get_pipeline_app():
    return create_pipeline()


pipeline_app = get_pipeline_app()

st.set_page_config(layout="wide")
st.title(
    "Quantifying Semantic Drift and Hallucination Accumulation in Sequential Multi-Agent LLM Architectures"
)
st.write("---")
st.header("Choose the dataset")

file_upload = st.file_uploader("Upload dev-v2 dataset", type=["json"])
if file_upload is not None:
    try:
        data_set = json.load(file_upload)
        records = data_set.get("data", [])

        all_data = []
        for record in records:
            title = record.get("title", "Unknown")
            for para in record.get("paragraphs", []):
                context_text = para.get("context", "")
                if not context_text:
                    continue
                for qa in para.get("qas", []):
                    answers_list = [a.get("text", "") for a in qa.get("answers", [])]
                    is_imp = qa.get("is_impossible", False)
                    gt_answer = answers_list[0] if answers_list else "UNANSWERABLE"

                    all_answers = answers_list if answers_list else ["UNANSWERABLE"]

                    all_data.append(
                        {
                            "source_text_id": f"{title}_{qa.get('id', '')}",
                            "topic": title,
                            "question": qa.get("question"),
                            "context": context_text,
                            "answers": answers_list,
                            "ground_truth_answer": gt_answer,
                            "all_valid_answers": all_answers,
                            "is_impossible": is_imp,
                        }
                    )
        all_data.sort(key=lambda r: r["source_text_id"])
        # print(all_data[:5])
        total_records = len(all_data)

        st.success(
            f"Dataset loaded Successfully! Total available records : {total_records}"
        )
        st.write("---")

        st.subheader("Experimental Configuration")
        col1, col2 = st.columns([1, 2])

        with col1:
            n_answerable = st.number_input(
                "Answerable items:",
                min_value=0,
                max_value=total_records,
                value=1,
                step=1,
            )
            n_unanswerable = st.number_input(
                "Unanswerable items:",
                min_value=0,
                max_value=total_records,
                value=1,
                step=1,
            )
            target_depth = st.selectbox("Pipeline Depth (n):", [1, 5, 10], index=0)
            exp_temp = st.selectbox(
                "Generation Temperature (T):", [0.1, 0.5, 0.9], index=1
            )
            random_seed = st.number_input("Random Seed:", min_value=1, value=42)
            generate_btn = st.button("Generate Experiment", type="primary")

        if generate_btn:
            run_started_at = datetime.datetime.now()
            run_id = run_started_at.strftime("%Y%m%dT%H%M%S")
            dataset_name = getattr(file_upload, "name", "unknown")
            random.seed(random_seed)

            answerable_pool = sorted(
                [r for r in all_data if not r["is_impossible"]],
                key=lambda r: r["source_text_id"],
            )
            unanswerable_pool = sorted(
                [r for r in all_data if r["is_impossible"]],
                key=lambda r: r["source_text_id"],
            )
            sample_answerable_items = random.sample(
                answerable_pool, min(n_answerable, len(answerable_pool))
            )
            sample_unanswerable_items = random.sample(
                unanswerable_pool, min(n_unanswerable, len(unanswerable_pool))
            )
            valid_sample = sample_answerable_items + sample_unanswerable_items

            st.write("---")
            st.header("Pipeline Execution")
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []
            rater_a_verdicts = []
            rater_b_verdicts = []

            for idx, row in enumerate(valid_sample):
                try:
                    status_text.text(
                        f"Processing item {idx + 1} of {len(valid_sample)}..."
                    )

                    status_text.text(
                        f"Processing {idx + 1}/{len(valid_sample)} - measuring n=0 baseline..."
                    )
                    baseline_understanding = extract_agent_understanding(
                        row["context"], row["question"], int(random_seed)
                    )
                    # st.write(baseline_understanding)
                    baseline_qa_loss = measure_f1_recoverability_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )

                    baseline_em_loss = measure_em_recoverability_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )
                    baseline_recall_loss = measure_qa_recall_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )
                    baseline_answer = baseline_understanding.get(
                        "answer", "CANNOT_ANSWER"
                    )
                    # st.write(f"Baseline QA loss: {baseline_qa_loss:.4f}")
                    # st.write(f"Baseline EM loss: {baseline_em_loss:.4f}")
                    # st.write(f"Baseline Recall loss: {baseline_recall_loss:.4f}")
                    # breakpoint()
                    initial_state = {
                        "original_context": row["context"],
                        "ground_truth_answer": row["ground_truth_answer"],
                        "current_text": row["context"],
                        "question": row["question"],
                        "is_impossible": row["is_impossible"],
                        "history": [],
                        "current_depth": 0,
                        "target_depth": target_depth,
                        "temperature": exp_temp,
                        "core_token_preservation_scores": [],
                        "seed": int(random_seed),
                        "semantic_drift_scores": [],
                        "global_drift_scores": [],
                        "qa_recoverability_scores": [],
                        "em_recoverability_scores": [],
                        "qa_recall_loss_scores": [],
                        "agent_answers": [],
                        "fabrication_flags": [],
                        "all_valid_answers": row["all_valid_answers"],
                    }
                    # st.write(initial_state)
                    # breakpoint()
                    progress_bar.progress((idx + 1) / len(valid_sample))

                    final_result = pipeline_app.invoke(initial_state)
                    scores = final_result.get("core_token_preservation_scores", [])
                    semantic_scores = final_result.get("semantic_drift_scores", [])
                    global_scores = final_result.get("global_drift_scores", [])
                    qa_recoverability_scores = final_result.get(
                        "qa_recoverability_scores", []
                    )
                    em_recoverability_scores = final_result.get(
                        "em_recoverability_scores", []
                    )
                    qa_recall_loss_scores = final_result.get(
                        "qa_recall_loss_scores", []
                    )
                    fabrication_flags = final_result.get("fabrication_flags", [])
                    agent_answers = final_result.get("agent_answers", [])

                    checkpoints = select_checkpoints(
                        len(final_result.get("history", []))
                    )
                    status_text.text(
                        f"Processing {idx + 1}/{len(valid_sample)} - rating faithfulness at hops {checkpoints}..."
                    )
                    checkpoint_verdicts = rate_faithfulness_at_checkpoints(
                        row["context"],
                        final_result.get("history", []),
                        row["is_impossible"],
                        int(random_seed),
                    )
                    final_hop = max(checkpoint_verdicts) if checkpoint_verdicts else 1
                    rater_a_faithful = checkpoint_verdicts.get(final_hop, {}).get(
                        "rater_a", 0
                    )
                    rater_b_faithful = checkpoint_verdicts.get(final_hop, {}).get(
                        "rater_b", 0
                    )

                    rater_a_verdicts.append(rater_a_faithful)
                    rater_b_verdicts.append(rater_b_faithful)

                    avg_score = float(np.mean(scores)) if scores else 0.0
                    final_score = float(scores[-1]) if scores else 0.0
                    avg_qa_recoverability = (
                        float(np.mean(qa_recoverability_scores))
                        if qa_recoverability_scores
                        else 0.0
                    )

                    local_drift_mean = (
                        float(np.mean(semantic_scores)) if semantic_scores else 0.0
                    )
                    local_drift_sum = (
                        float(np.sum(semantic_scores)) if semantic_scores else 0.0
                    )
                    global_drift_final = (
                        float(global_scores[-1]) if global_scores else 0.0
                    )
                    global_drift_max = (
                        float(np.max(global_scores)) if global_scores else 0.0
                    )
                    drift_cancellation = local_drift_sum - global_drift_final

                    final_qa_loss = (
                        float(qa_recoverability_scores[-1])
                        if qa_recoverability_scores
                        else 0.0
                    )
                    attributable_qa_loss = max(0.0, final_qa_loss - baseline_qa_loss)
                    baseline_recoverable = 1 if baseline_qa_loss == 0.0 else 0

                    final_em_loss = (
                        float(em_recoverability_scores[-1])
                        if em_recoverability_scores
                        else 0.0
                    )
                    attributable_em_loss = max(0.0, final_em_loss - baseline_em_loss)

                    final_recall_loss = (
                        float(qa_recall_loss_scores[-1])
                        if qa_recall_loss_scores
                        else 0.0
                    )
                    attributable_recall_loss = max(
                        0.0, final_recall_loss - baseline_recall_loss
                    )

                    em_f1_gap = attributable_em_loss - attributable_qa_loss

                    if row["is_impossible"]:
                        n_fab = sum(fabrication_flags)
                        n_hops = len(fabrication_flags)
                        if n_hops == 0:
                            failure_mode = "No hops recorded"
                        elif n_fab == 0:
                            failure_mode = "Correctly refused to answer (faithful)"
                        elif n_fab == n_hops:
                            failure_mode = "Fabrication (hallucination)"
                        else:
                            failure_mode = "Drift-Induced Fabrication (some hops)"
                    elif rater_a_faithful == 1 and rater_b_faithful == 1:
                        if global_drift_final > SEMANTIC_SHIFT_THRESHOLD:
                            failure_mode = "High Semantic Shift (no hallucination)"
                        else:
                            failure_mode = "Stable / Conserved"
                    else:
                        failure_mode = "Hallucination / Faithfulness Failure"

                    results_list.append(
                        {
                            "sample_id": row["source_text_id"],
                            "topic": row["topic"],
                            "question": row["question"],
                            "ground_truth": row["ground_truth_answer"],
                            "all_valid_answers": str(row["all_valid_answers"]),
                            "original_text": (
                                row["context"][:150] + "..."
                                if len(row["context"]) > 150
                                else row["context"]
                            ),
                            "final_text": (
                                final_result["current_text"][:150] + "..."
                                if len(final_result["current_text"]) > 150
                                else final_result["current_text"]
                            ),
                            "original_text_full": row["context"],
                            "final_text_full": final_result["current_text"],
                            "text_history": final_result["history"],
                            "scores_history": scores,
                            "semantic_scores_history": semantic_scores,
                            "global_drift_history": global_scores,
                            "global_drift_final": global_drift_final,
                            "global_drift_max": global_drift_max,
                            "local_drift_mean": local_drift_mean,
                            "local_drift_sum": local_drift_sum,
                            "drift_cancellation": drift_cancellation,
                            "qa_recoverability_history": qa_recoverability_scores,
                            "pipeline_depth": target_depth,
                            "temperature": exp_temp,
                            "iterations_completed": final_result["current_depth"],
                            "avg_core_token_preservation": avg_score,
                            "final_core_token_preservation": final_score,
                            "avg_lexical_loss": 1.0 - avg_score,
                            "final_lexical_loss": 1.0 - final_score,
                            "baseline_qa_loss": baseline_qa_loss,
                            "baseline_answer": baseline_answer,
                            "baseline_recoverable": baseline_recoverable,
                            "final_qa_loss": final_qa_loss,
                            "attributable_qa_loss": attributable_qa_loss,
                            "baseline_recall_loss": baseline_recall_loss,
                            "final_recall_loss": final_recall_loss,
                            "attributable_recall_loss": attributable_recall_loss,
                            "qa_recall_loss_history": qa_recall_loss_scores,
                            "fabrication_history": fabrication_flags,
                            "agent_answers_history": str(agent_answers),
                            "fabricated_any_hop": int(any(fabrication_flags)),
                            "first_fabrication_hop": (
                                fabrication_flags.index(1) + 1
                                if 1 in fabrication_flags
                                else 0
                            ),
                            "fabrication_rate": (
                                float(np.mean(fabrication_flags))
                                if fabrication_flags
                                else 0.0
                            ),
                            "checkpoint_hops": list(checkpoint_verdicts.keys()),
                            "checkpoint_both_faithful": [
                                v["both_faithful"] for v in checkpoint_verdicts.values()
                            ],
                            "baseline_em_loss": baseline_em_loss,
                            "final_em_loss": final_em_loss,
                            "attributable_em_loss": attributable_em_loss,
                            "em_recoverability_history": em_recoverability_scores,
                            "em_f1_gap": em_f1_gap,
                            "avg_qa_recoverability": avg_qa_recoverability,
                            "global_drift_similarity": 1.0 - global_drift_final,
                            "rater_a_faithful": rater_a_faithful,
                            "rater_b_faithful": rater_b_faithful,
                            "run_id": run_id,
                            "run_timestamp": run_started_at.isoformat(
                                timespec="seconds"
                            ),
                            "dataset_file": dataset_name,
                            "dataset": "SQuAD v2.0",
                            "sample_size_n": len(valid_sample),
                            "random_seed": int(random_seed),
                            "agent_model": AGENT_MODEL,
                            "reader_model": READER_MODEL,
                            "rater_a_model": RATER_MODEL_A,
                            "rater_b_model": RATER_MODEL_B,
                            "rater_a_seed": int(random_seed),
                            "rater_b_seed": int(random_seed),
                            "embedder_model": EMBEDDER_MODEL,
                            "agent_temperature": exp_temp,
                            "rater_temperature": 0.0,
                            "reader_temperature": 0.0,
                            "semantic_shift_threshold": SEMANTIC_SHIFT_THRESHOLD,
                            "is_impossible": row["is_impossible"],
                            "qa_metric": "SQuAD official EM + token-F1 (max over gold answers)",
                            "diagnosed_cause": failure_mode,
                            "is_hallucination_failure": (
                                # 1 if failure_mode.startswith("Hallucination") else 0
                                1
                                if (row["is_impossible"] and any(fabrication_flags))
                                or (
                                    not row["is_impossible"]
                                    and not (
                                        rater_a_faithful == 1 and rater_b_faithful == 1
                                    )
                                )
                                else 0
                            ),
                            "rater_agreement": (
                                1 if rater_a_faithful == rater_b_faithful else 0
                            ),
                        }
                    )
                    try:
                        pd.DataFrame(results_list).to_csv(
                            f"_drift_d{pipeline_depth}_T{temperature}_seed{random_seed}_PART.csv",
                            index=False,
                        )
                    except Exception:
                        pass
                    progress_bar.progress(min(1.0, (idx + 1) / len(valid_sample)))
                except Exception as e:
                    st.error(f"Error processing item {idx + 1}: {e}")
                    continue

            status_text.empty()
            progress_bar.empty()

            st.session_state["results_df"] = pd.DataFrame(results_list)
            st.success("Pipeline execution complete!")
            st.rerun()

        if "results_df" in st.session_state:
            df = st.session_state["results_df"]

            rater_a_verdicts = df["rater_a_faithful"].tolist()
            rater_b_verdicts = df["rater_b_faithful"].tolist()

            cohens_kappa = (
                cohen_kappa_score(rater_a_verdicts, rater_b_verdicts)
                if rater_a_verdicts
                else 0.0
            )

            agreement_list = [
                1 if v1 == v2 else 0
                for v1, v2 in zip(rater_a_verdicts, rater_b_verdicts)
            ]
            avg_agreement_pct = (
                (sum(agreement_list) / len(agreement_list)) * 100
                if agreement_list
                else 0.0
            )

            if cohens_kappa is None or math.isnan(cohens_kappa):
                cohens_kappa = None
                kappa_calculated = False
            else:
                kappa_calculated = True

            st.header("Pipeline Results")
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric("Samples Processed", len(df))
            with col_m2:
                st.metric(
                    "Pipeline Depth",
                    int(df["pipeline_depth"].iloc[0]) if len(df) > 0 else 0,
                )
            with col_m3:
                st.metric(
                    "Temperature",
                    float(df["temperature"].iloc[0]) if len(df) > 0 else 0,
                )
            st.write("---")

            col_m4, col_m5, col_m6 = st.columns(3)

            with col_m4:
                st.metric(
                    "Avg Core-Token Preservation",
                    f"{df['avg_core_token_preservation'].mean():.2%}",
                )
            with col_m5:
                st.metric(
                    "Avg Global Semantic Drift",
                    f"{df['global_drift_final'].mean():.2%}",
                )
            with col_m6:
                st.metric(
                    "Avg QA Recoverability Loss",
                    f"{df['avg_qa_recoverability'].mean():.2%}",
                )

            st.write("---")

            col_m8, col_m9, col_m10 = st.columns(3)

            with col_m8:
                st.metric("Rater A-B Agreement", f"{df['rater_agreement'].mean():.2%}")
            with col_m9:
                if kappa_calculated:
                    st.metric(
                        "Cohen's κ (inter-rater, faithfulness)", f"{cohens_kappa:.4f}"
                    )
                else:
                    st.metric("Cohen's κ (inter-rater, faithfulness)", "n/a")
            with col_m10:
                hallucination_rate = (
                    df["is_hallucination_failure"].mean() if len(df) > 0 else 0.0
                )
                st.metric("Hallucination Rate", f"{hallucination_rate:.1%}")

            st.markdown("#### Hallucination Accumulation by Depth")

            imp_df = (
                df[df["is_impossible"] == True]
                if "is_impossible" in df.columns
                else df.iloc[0:0]
            )
            try:
                if len(imp_df) > 0:
                    max_hops = int(imp_df["iterations_completed"].max())
                    fab_rows = []
                    for hop in range(max_hops):
                        vals = [
                            h[hop]
                            for h in imp_df["fabrication_history"]
                            if isinstance(h, list) and len(h) > hop
                        ]
                        if vals:
                            fab_rows.append(
                                {
                                    "Hop": hop + 1,
                                    "Fabrication Rate (%)": float(np.mean(vals)) * 100,
                                }
                            )
                    if fab_rows:
                        st.line_chart(
                            pd.DataFrame(fab_rows).set_index("Hop"), color=["#dc3545"]
                        )
                        st.caption(
                            "Proportion of UNANSWERABLE items where the agent invented an answer "
                            "instead of refusing, measured at each hop. This is a ground-truth-anchored "
                            "hallucination measure: it depends on SQuAD's is_impossible label, not on "
                            "another model's opinion."
                        )
                    h1, h2 = st.columns(2)
                    with h1:
                        st.metric(
                            "Items Fabricated at Some Hop",
                            f"{imp_df['fabricated_any_hop'].mean():.1%}",
                        )
                    with h2:
                        first = imp_df[imp_df["first_fabrication_hop"] > 0][
                            "first_fabrication_hop"
                        ]
                        st.metric(
                            "Mean First Fabrication Hop",
                            f"{first.mean():.1f}" if len(first) else "n/a",
                            help="How deep into the chain hallucination typically first appears.",
                        )
                else:
                    st.info(
                        "No unanswerable (is_impossible) items in this sample, so fabrication rate "
                        "cannot be computed. Include unanswerable items to measure hallucination "
                        "accumulation."
                    )

                cp_rows = {}
                for hops, faiths in zip(
                    df["checkpoint_hops"], df["checkpoint_both_faithful"]
                ):
                    if isinstance(hops, list) and isinstance(faiths, list):
                        for h, f in zip(hops, faiths):
                            cp_rows.setdefault(h, []).append(f)
                if cp_rows:
                    cp_df = pd.DataFrame(
                        [
                            {"Hop": h, "Hallucination Rate (%)": (1 - np.mean(v)) * 100}
                            for h, v in sorted(cp_rows.items())
                        ]
                    ).set_index("Hop")
                    st.line_chart(cp_df, color=["#fd7e14"])
                    st.caption(
                        "Rater-judged hallucination rate at checkpoint hops (both raters must agree "
                        "the text is faithful for it to count as clean)."
                    )
            except Exception as e:
                st.warning(f"Could not render hallucination accumulation: {e}")

            st.markdown("#### Answer Recovery: Exact Match vs Token-F1")
            try:
                max_h = int(df["iterations_completed"].max())
                rows = []
                for hop in range(max_h):
                    f1v = [
                        r[hop]
                        for r in df["qa_recoverability_history"]
                        if isinstance(r, list) and len(r) > hop
                    ]
                    emv = [
                        r[hop]
                        for r in df["em_recoverability_history"]
                        if isinstance(r, list) and len(r) > hop
                    ]
                    if f1v or emv:
                        rows.append(
                            {
                                "Hop": hop + 1,
                                "EM loss (strict) %": (
                                    (float(np.mean(emv)) * 100) if emv else None
                                ),
                                "F1 loss (lenient) %": (
                                    (float(np.mean(f1v)) * 100) if f1v else None
                                ),
                            }
                        )
                if rows:
                    st.line_chart(pd.DataFrame(rows).set_index("Hop"))

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.metric(
                        "Attributable EM Loss",
                        f"{df['attributable_em_loss'].mean():.1%}",
                    )
                with g2:
                    st.metric(
                        "Attributable F1 Loss",
                        f"{df['attributable_qa_loss'].mean():.1%}",
                    )
                with g3:
                    st.metric(
                        "EM-F1 Gap",
                        f"{df['em_f1_gap'].mean():.1%}",
                    )

                gap = df["em_f1_gap"].mean()
                if gap > 0.15:
                    st.info(f"EM degraded {gap:.0%} more than F1.")
                elif len(df) > 0:
                    st.info(
                        "EM and F1 degraded together, indicating the answer content was lost "
                    )
            except Exception as e:
                st.warning(f"Could not render EM vs F1 comparison: {e}")

            st.markdown("### Statistical Summary")

            st.markdown("#### Recoverability vs the n=0 Baseline")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric(
                    "Reader Floor (n=0)",
                    f"{df['baseline_qa_loss'].mean():.1%}",
                )
            with b2:
                st.metric("Total Loss (final hop)", f"{df['final_qa_loss'].mean():.1%}")
            with b3:
                st.metric(
                    "Attributable to Pipeline",
                    f"{df['attributable_qa_loss'].mean():.1%}",
                )
            if df["baseline_recoverable"].mean() < 0.6:
                st.warning(
                    f"The reader answered only {df['baseline_recoverable'].mean():.0%} of questions correctly from the ORIGINAL context."
                )

            st.markdown("#### Drift Accumulation Curve (Local vs Global)")
            try:
                max_hops = int(df["iterations_completed"].max())
                if max_hops > 0:
                    curve_rows = []
                    for hop in range(max_hops):
                        g_vals = [
                            r[hop]
                            for r in df["global_drift_history"]
                            if isinstance(r, list) and len(r) > hop
                        ]
                        l_vals = [
                            r[hop]
                            for r in df["semantic_scores_history"]
                            if isinstance(r, list) and len(r) > hop
                        ]
                        curve_rows.append(
                            {
                                "Hop": hop + 1,
                                "Global drift (vs original)": (
                                    float(np.mean(g_vals)) if g_vals else None
                                ),
                                "Local drift (vs previous hop)": (
                                    float(np.mean(l_vals)) if l_vals else None
                                ),
                            }
                        )
                    curve_df = pd.DataFrame(curve_rows).set_index("Hop")
                    st.line_chart(curve_df)
                    st.caption(
                        "Local drift is measured hop-to-hop and stays flat by construction. "
                        "Global drift is measured against the original source and is the primary "
                        "metric: it is what reveals cumulative degradation with pipeline depth."
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Global Drift (final hop)",
                            f"{df['global_drift_final'].mean():.2%}",
                        )
                    with c2:
                        st.metric(
                            "Cumulative Local Drift",
                            f"{df['local_drift_sum'].mean():.2%}",
                        )
                    with c3:
                        st.metric(
                            "Drift Cancellation",
                            f"{df['drift_cancellation'].mean():.2%}",
                        )
            except Exception as e:
                st.warning(f"Could not render drift curve: {e}")

            stat_cols = st.columns(2)

            with stat_cols[0]:
                st.write("**Rater A (faithfulness)**")
                j1_mean = df["rater_a_faithful"].mean()
                st.metric("Judged Faithful", f"{j1_mean:.1%}")

            with stat_cols[1]:
                st.write("**Rater B (faithfulness)**")
                j2_mean = df["rater_b_faithful"].mean()
                st.metric("Judged Faithful", f"{j2_mean:.1%}")

            st.write("---")
            st.subheader("Agent History & Audit Viewer")
            st.write(
                "Select a sample below to inspect exact propagation hops and ground-truth alignment:"
            )

            selected_sample_id = st.selectbox(
                "Select Sample ID:", options=df["sample_id"].tolist()
            )

            if selected_sample_id:
                sample_data = df[df["sample_id"] == selected_sample_id].iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Topic:** {sample_data['topic']}")
                with c2:
                    st.markdown(
                        f"**Rater A (faithfulness):** {'Faithful' if sample_data['rater_a_faithful'] == 1 else 'Hallucinated'}"
                    )
                with c3:
                    st.markdown(
                        f"**Rater B (faithfulness):** {'Faithful' if sample_data['rater_b_faithful'] == 1 else 'Hallucinated'}"
                    )

                st.markdown("### Benchmark Reference Data")
                st.info(
                    f"**Question Asked:** {sample_data['question']}\n\n**Gold Standard Answer:** {sample_data['ground_truth']}\n\n**All Valid Answers:** {sample_data['all_valid_answers']}"
                )

                with st.expander(
                    "Background SQuAD Context (Input at Hop 0)", expanded=False
                ):
                    st.write(sample_data["original_text_full"])

                st.markdown("### Agent Chain Propagation History")
                text_history = sample_data["text_history"]
                semantic_history = sample_data["semantic_scores_history"]
                qa_history = sample_data["qa_recoverability_history"]
                # answers_history = ast.literal_eval(sample_data["agent_answers_history"])
                answers_history = ast.literal_eval(
                    sample_data.get("agent_answers_history", "[]")
                )

                if isinstance(semantic_history, str):
                    semantic_history = ast.literal_eval(semantic_history)
                if isinstance(qa_history, str):
                    qa_history = ast.literal_eval(qa_history)
                if isinstance(answers_history, str):
                    answers_history = ast.literal_eval(answers_history)

                qa_labels = {0.0: "Preserved", 0.5: "Partial Loss", 1.0: "Lost"}

                for i, text in enumerate(text_history):
                    depth = i + 1
                    drift = semantic_history[i] if i < len(semantic_history) else 0.0
                    similarity = 1.0 - drift
                    qa_score = qa_history[i] if i < len(qa_history) else 0.0
                    qa_label = qa_labels.get(qa_score, f"{qa_score:.1f}")

                    is_expanded = depth == len(text_history)
                    expander_title = (
                        f"Hop {depth} Output — Semantic Similarity: {similarity:.1%} | "
                        f"Semantic Drift: {drift:.1%} | Answer Recoverability: {qa_label}"
                    )
                    if depth == len(text_history):
                        expander_title = "FINAL OUTPUT — " + expander_title

                    with st.expander(expander_title, expanded=is_expanded):
                        st.write(text)
                        reader_ans = (
                            answers_history[i] if i < len(answers_history) else "N/A"
                        )
                        if sample_data["is_impossible"]:
                            verdict = (
                                "Correctly refused"
                                if "CANNOT_ANSWER" in str(reader_ans).upper()
                                else "Fabricated"
                            )
                        else:
                            verdict = ""
                        st.caption(f"Reader Answer: {reader_ans} | Verdict: {verdict}")

                st.subheader("Information Degradation Curves")

                semantic_list = semantic_history
                chart_data = pd.DataFrame(
                    {
                        "Hop": range(1, len(semantic_list) + 1),
                        "Local Semantic Drift (%)": [s * 100 for s in semantic_list],
                    }
                ).set_index("Hop")
                st.line_chart(chart_data, color=["#ffc107"])
                st.caption(
                    "Local (hop-to-hop) embedding drift. See the Global Drift curve for cumulative divergence from the source."
                )

                qa_chart_data = pd.DataFrame(
                    {
                        "Hop": range(1, len(qa_history) + 1),
                        "Answer Recoverability Loss (%)": [s * 100 for s in qa_history],
                    }
                ).set_index("Hop")
                st.line_chart(qa_chart_data, color=["#dc3545"])

            st.markdown("---")

            st.markdown("### Inter-Rater Agreement Analysis")
            col_j_agree, col_failure = st.columns(2)

            with col_j_agree:
                st.write("**Rater Verdict Agreement (A vs B)**")
                agree = df["rater_a_faithful"] == df["rater_b_faithful"]

                agreement_counts = [agree.sum(), (~agree).sum()]
                agreement_labels = ["Agree", "Disagree"]

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=agreement_labels,
                            values=agreement_counts,
                            marker=dict(colors=["#28a745", "#dc3545"]),
                        )
                    ]
                )
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            with col_failure:
                st.write("**Failure Mode Distribution**")
                failure_counts = df["diagnosed_cause"].value_counts()

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=failure_counts.index,
                            values=failure_counts.values,
                            marker=dict(
                                colors=["#0d6efd", "#198754", "#fd7e14", "#6c757d"]
                            ),
                        )
                    ]
                )
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Rater Verdict Heatmap (Sample x Rater Agreement)")

            heatmap_df = df[
                ["topic", "rater_a_faithful", "rater_b_faithful", "diagnosed_cause"]
            ].copy()
            heatmap_df["All_Agree"] = (
                heatmap_df["rater_a_faithful"] == heatmap_df["rater_b_faithful"]
            ).astype(int)
            heatmap_df = heatmap_df.rename(
                columns={
                    "rater_a_faithful": "Rater A (Faithful)",
                    "rater_b_faithful": "Rater B (Faithful)",
                    "All_Agree": "All Agree",
                    "diagnosed_cause": "Failure Mode",
                }
            )

            st.dataframe(
                heatmap_df.style.map(
                    lambda x: (
                        "background-color: #28a745; color: white"
                        if x == 1
                        else "background-color: #dc3545; color: white" if x == 0 else ""
                    )
                ),
                use_container_width=True,
                height=400,
            )

            st.markdown("---")
            st.subheader("Detailed Results")
            display_cols = [
                "sample_id",
                "topic",
                "question",
                "ground_truth",
                "final_text",
                "avg_core_token_preservation",
                "global_drift_final",
                "avg_qa_recoverability",
                "rater_a_faithful",
                "rater_b_faithful",
                "diagnosed_cause",
            ]
            st.dataframe(df[display_cols], use_container_width=True, height=400)

            st.write("---")
            st.subheader("Export Results")
            run_tag = (
                df["run_id"].iloc[0] if "run_id" in df.columns and len(df) else "run"
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Pipeline Results (CSV)",
                data=csv,
                file_name=(
                    f"drift_d{target_depth}_T{exp_temp}_seed{int(random_seed)}_{run_tag}.csv"
                ),
                mime="text/csv",
            )

            with st.expander("Run manifest (provenance recorded in this CSV)"):
                manifest_cols = [
                    "run_id",
                    "run_timestamp",
                    "dataset",
                    "dataset_file",
                    "sample_size_n",
                    "pipeline_depth",
                    "agent_temperature",
                    "random_seed",
                    "agent_model",
                    "reader_model",
                    "rater_a_model",
                    "rater_b_model",
                    "embedder_model",
                    "qa_metric",
                ]
                present = [c for c in manifest_cols if c in df.columns]
                st.table(df[present].iloc[0].rename("value").to_frame().astype(str))

    except Exception as e:
        st.error(f"Error processing file: {e}")
        import traceback

        st.error(traceback.format_exc())

else:
    st.info("Please upload the SQuAD dev-v2 dataset (JSON) to begin")
