import random
import streamlit as st
import json
import pandas as pd
import numpy as np
import math
import ast
from sklearn.metrics import cohen_kappa_score
import datetime
from configuration import (
    AGENT_MODEL,
    READER_MODEL,
    RATER_MODEL_A,
    RATER_MODEL_B,
    EMBEDDER_MODEL,
)
from reader import extract_agent_understanding
from rater import rate_faithfulness_at_checkpoints, select_checkpoints
from pipeline import (
    create_pipeline,
    # extract_agent_understanding,
    # rate_faithfulness_at_checkpoints,
    # select_checkpoints,
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
        # st.write(all_data[:135])

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
                value=15,
                step=1,
            )
            n_unanswerable = st.number_input(
                "Unanswerable items:",
                min_value=0,
                max_value=total_records,
                value=15,
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

            answerable_pool = []
            for r in all_data:
                if r["is_impossible"] == False:
                    answerable_pool.append(r)

                def get_sort_key(record):
                    return record["source_text_id"]

                answerable_pool.sort(key=get_sort_key)

            unanswerable_pool = []
            for r in all_data:
                if r["is_impossible"] == True:
                    unanswerable_pool.append(r)

                def get_sort_key(record):
                    return record["source_text_id"]

                unanswerable_pool.sort(key=get_sort_key)

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
                    # st.write("baseline_understanding ", baseline_understanding)
                    status_text.text(
                        f"Baseline understanding accuired!, measuring Baseline answer recoverability loss.."
                    )
                    baseline_qa_loss = measure_f1_recoverability_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )
                    # st.write("baseline_qa_loss ", baseline_qa_loss)
                    status_text.text(
                        f"Baseline answer recoverability accuired!, measuring Baseline em recoverability loss.."
                    )
                    baseline_em_loss = measure_em_recoverability_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )
                    # st.write("baseline_em_loss", baseline_em_loss)
                    status_text.text(
                        f"Baseline em recoverability loss accuired!, measuring Baseline recall loss.."
                    )
                    baseline_recall_loss = measure_qa_recall_loss(
                        baseline_understanding,
                        row["all_valid_answers"],
                        row["is_impossible"],
                    )
                    # st.write("baseline_recall_loss ", baseline_recall_loss)
                    status_text.text(
                        f"Baseline recall loss accuired!, running the pipeline..{idx + 1}/{len(valid_sample)}"
                    )
                    baseline_answer = baseline_understanding.get(
                        "answer", "CANNOT_ANSWER"
                    )
                    # st.write("baseline_answer ", baseline_answer)
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
                    # st.write(final_result)
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
                    if checkpoint_verdicts:
                        final_hop = max(checkpoint_verdicts)
                    else:
                        final_hop = 1

                    rater_a_faithful = checkpoint_verdicts.get(final_hop, {}).get(
                        "rater_a", 0
                    )
                    rater_b_faithful = checkpoint_verdicts.get(final_hop, {}).get(
                        "rater_b", 0
                    )

                    rater_a_verdicts.append(rater_a_faithful)
                    rater_b_verdicts.append(rater_b_faithful)

                    if scores:
                        avg_score = float(np.mean(scores))
                    else:
                        avg_score = 0.0

                    if scores:
                        final_score = float(scores[-1])
                    else:
                        final_score = 0.0

                    if qa_recoverability_scores:
                        avg_qa_recoverability = float(np.mean(qa_recoverability_scores))
                    else:
                        avg_qa_recoverability = 0.0

                    if semantic_scores:
                        local_drift_mean = float(np.mean(semantic_scores))
                    else:
                        local_drift_mean = 0.0

                    if semantic_scores:
                        local_drift_sum = float(np.sum(semantic_scores))
                    else:
                        local_drift_sum = 0.0

                    if global_scores:
                        global_drift_final = float(global_scores[-1])
                    else:
                        global_drift_final = 0.0

                    if global_scores:
                        global_drift_max = float(np.max(global_scores))
                    else:
                        global_drift_max = 0.0

                    if qa_recoverability_scores:
                        final_qa_loss = float(qa_recoverability_scores[-1])
                    else:
                        final_qa_loss = 0.0

                    attributable_qa_loss = max(0.0, final_qa_loss - baseline_qa_loss)
                    if baseline_qa_loss == 0.0:
                        baseline_recoverable = 1
                    else:
                        baseline_recoverable = 0

                    if em_recoverability_scores:
                        final_em_loss = float(em_recoverability_scores[-1])
                    else:
                        final_em_loss = 0.0

                    attributable_em_loss = max(0.0, final_em_loss - baseline_em_loss)

                    if qa_recall_loss_scores:
                        final_recall_loss = float(qa_recall_loss_scores[-1])
                    else:
                        final_recall_loss = 0.0

                    attributable_recall_loss = max(
                        0.0, final_recall_loss - baseline_recall_loss
                    )

                    em_f1_gap = attributable_em_loss - attributable_qa_loss

                    if (row["is_impossible"] and any(fabrication_flags)) or (
                        not row["is_impossible"]
                        and not (rater_a_faithful == 1 and rater_b_faithful == 1)
                    ):
                        _is_hallucination_failure_res = 1
                    else:
                        _is_hallucination_failure_res = 0

                    if fabrication_flags:
                        _fab_rication_rate = float(np.mean(fabrication_flags))
                    else:
                        _fab_rication_rate = 0.0

                    checkpoint_both_faithfull = []
                    for v in checkpoint_verdicts.values():
                        checkpoint_both_faithfull.append(v["both_faithful"])

                    if rater_a_faithful == rater_b_faithful:
                        rater_agreements = 1
                    else:
                        rater_agreements = 0

                    if len(final_result["current_text"]) > 150:
                        fin_tex = final_result["current_text"][:150] + "..."
                    else:
                        fin_tex = final_result["current_text"]

                    if len(row["context"]) > 150:
                        ori_txt = row["context"][:150] + "..."
                    else:
                        ori_txt = row["context"]

                    results_list.append(
                        {
                            "sample_id": row["source_text_id"],
                            "topic": row["topic"],
                            "question": row["question"],
                            "ground_truth": row["ground_truth_answer"],
                            "all_valid_answers": str(row["all_valid_answers"]),
                            "original_text": ori_txt,
                            "final_text": fin_tex,
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
                            "qa_recoverability_history": qa_recoverability_scores,
                            "pipeline_depth": target_depth,
                            "temperature": exp_temp,
                            "iterations_completed": final_result["current_depth"],
                            "avg_core_token_preservation": avg_score,
                            "final_core_token_preservation": final_score,
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
                            "first_fabrication_hop": _is_hallucination_failure_res,
                            "fabrication_rate": _fab_rication_rate,
                            "checkpoint_hops": list(checkpoint_verdicts.keys()),
                            "checkpoint_both_faithful": checkpoint_both_faithfull,
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
                            "is_impossible": row["is_impossible"],
                            "is_hallucination_failure": _is_hallucination_failure_res,
                            "rater_agreement": rater_agreements,
                        }
                    )

                    try:
                        pd.DataFrame(results_list).to_csv(
                            f"_drift_d{target_depth}_T{exp_temp}_seed{random_seed}_PART.csv",
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

            if rater_a_verdicts:
                cohens_kappa = cohen_kappa_score(rater_a_verdicts, rater_b_verdicts)
            else:
                cohens_kappa = 0.0

            agreement_list = []
            for v1, v2 in zip(rater_a_verdicts, rater_b_verdicts):
                if v1 == v2:
                    agreement_list.append(1)
                else:
                    agreement_list.append(0)

            if agreement_list:
                avg_agreement_pct = (sum(agreement_list) / len(agreement_list)) * 100
            else:
                avg_agreement_pct = 0.0

            if cohens_kappa is None or math.isnan(cohens_kappa):
                cohens_kappa = None
                kappa_calculated = False
            else:
                kappa_calculated = True

            st.header("Pipeline Results")
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric("Samples Processed", len(df))

            if len(df) > 0:
                dpth = int(df["pipeline_depth"].iloc[0])
            else:
                dpth = 0
            with col_m2:
                st.metric(
                    "Pipeline Depth",
                    dpth,
                )

            if len(df) > 0:
                tmps = float(df["temperature"].iloc[0])
            else:
                tmps = 0
            with col_m3:
                st.metric(
                    "Temperature",
                    tmps,
                )
            st.write("---")
            st.markdown("### Statistical Summary")

            b1, b2 = st.columns(2)
            with b1:
                st.metric(
                    "Reader Floor (n=0)",
                    f"{df['baseline_qa_loss'].mean():.1%}",
                )

            with b2:
                st.metric(
                    "Attributable loss to Pipeline",
                    f"{df['attributable_qa_loss'].mean():.1%}",
                )
            st.write("---")
            col_m4, col_m5, col_m6 = st.columns(3)

            with col_m4:
                st.metric(
                    "Avg Core-Token Preservation",
                    f"{df['avg_core_token_preservation'].mean():.2%}",
                )

            st.write("---")
            st.markdown("#### Core token preservation")
            try:
                tot_hop = int(df["iterations_completed"].max())
                ct_row = []
                for hop in range(tot_hop):
                    ct_va = []
                    for r in df["scores_history"]:
                        if isinstance(r, list) and len(r) > hop:
                            ct_va.append(r[hop])

                    if ct_va:
                        ct_row.append(
                            {
                                "Hop": hop + 1,
                                "Core Token Preservation": float(np.mean(ct_va)) * 100,
                            }
                        )

                if ct_row:
                    st.line_chart(
                        pd.DataFrame(ct_row).set_index("Hop"), color=["#dc3545"]
                    )

            except Exception as e:
                st.write(f"Error: {e}")

            st.write("---")
            st.markdown("#### Semantic Drift")
            cos1, cos2 = st.columns(2)
            with cos1:
                st.metric(
                    "Avg Global Semantic Drift",
                    f"{df['global_drift_final'].mean():.2%}",
                )
            with cos2:
                st.metric(
                    "Avg Local Semantic Drift", f"{df['local_drift_mean'].mean():.2%}"
                )

            st.write("Drift Accumulation Curve (Local vs Global)")
            try:
                max_hops = int(df["iterations_completed"].max())
                if max_hops <= 0:
                    raise ValueError("No completed iterations found.")

                curve_rows = []
                for hop in range(max_hops):
                    g_vals = []
                    for r in df["global_drift_history"]:
                        if isinstance(r, list) and len(r) > hop:
                            g_vals.append(r[hop])

                    l_vals = []
                    for r in df["semantic_scores_history"]:
                        if isinstance(r, list) and len(r) > hop:
                            l_vals.append(r[hop])

                    if g_vals:
                        glo_d_mean = float(np.mean(g_vals))
                    else:
                        glo_d_mean = None

                    if l_vals:
                        loc_d_mean = float(np.mean(l_vals))
                    else:
                        loc_d_mean = None

                    curve_rows.append(
                        {
                            "Hop": hop + 1,
                            "Global drift (vs original)": glo_d_mean * 100,
                            "Local drift (vs previous hop)": loc_d_mean * 100,
                        }
                    )

                st.line_chart(pd.DataFrame(curve_rows).set_index("Hop"))

            except Exception as e:
                st.warning(f"Error: {e}")
            st.write("---")

            st.markdown("#### Answer Recovery")
            g1, g2, g3, g4, g5, g6 = st.columns(6)

            with g1:
                st.metric(
                    "Avg QA Recoverability Loss",
                    f"{df['avg_qa_recoverability'].mean():.2%}",
                )
            with g2:
                st.metric(
                    "Attributable EM Loss",
                    f"{df['attributable_em_loss'].mean():.1%}",
                )
            with g3:
                st.metric(
                    "Attributable F1 Loss", f"{df['attributable_qa_loss'].mean():.1%}"
                )
            with g4:
                st.metric(
                    "EM-F1 Gap",
                    f"{df['em_f1_gap'].mean():.1%}",
                )
            ans_df = df[df["is_impossible"] == False]
            with g5:
                st.metric("Total Loss", f"{ans_df['final_qa_loss'].mean():.1%}")

            with g6:
                st.metric("Avg Recall Loss", f"{df['final_recall_loss'].mean():.1%}")
            gap = df["em_f1_gap"].mean()
            if gap > 0.15:
                st.info(f"EM degraded {gap:.0%} more than F1.")
            elif len(df) > 0:
                st.info(
                    "EM and F1 degraded together, indicating the answer content was lost "
                )

            st.write("Exact Match vs Token-F1 vs Recall Loss")
            try:
                max_h = int(df["iterations_completed"].max())
                rows = []

                for hop in range(max_h):
                    f1v = []
                    for r in df["qa_recoverability_history"]:
                        if isinstance(r, str):
                            r = ast.literal_eval(r)
                        if isinstance(r, list) and len(r) > hop:
                            f1v.append(r[hop])

                    emv = []
                    for r in df["em_recoverability_history"]:
                        if isinstance(r, str):
                            r = ast.literal_eval(r)
                        if isinstance(r, list) and len(r) > hop:
                            emv.append(r[hop])

                    recv = []
                    for r in df["qa_recall_loss_history"]:
                        if isinstance(r, str):
                            r = ast.literal_eval(r)
                        if isinstance(r, list) and len(r) > hop:
                            recv.append(r[hop])

                    if f1v or emv or recv:
                        _f1_loss_pct = (float(np.mean(f1v)) * 100) if f1v else None
                        _em_loss_pct = (float(np.mean(emv)) * 100) if emv else None
                        _recall_loss_pct = (
                            (float(np.mean(recv)) * 100) if recv else None
                        )

                        rows.append(
                            {
                                "Hop": hop + 1,
                                "EM loss (strict) %": _em_loss_pct,
                                "F1 loss (lenient) %": _f1_loss_pct,
                                "Recall loss %": _recall_loss_pct,
                            }
                        )

                if rows:
                    st.line_chart(pd.DataFrame(rows).set_index("Hop"))

            except Exception as e:
                st.warning(f"Error: {e}")

            st.write("---")
            st.markdown("### Faithfulness")
            stat_cols = st.columns(5)

            with stat_cols[0]:
                st.write("**Rater A (faithfulness)**")
                j1_mean = df["rater_a_faithful"].mean()
                st.metric("Judged Faithful", f"{j1_mean:.1%}")

            with stat_cols[1]:
                st.write("**Rater B (faithfulness)**")
                j2_mean = df["rater_b_faithful"].mean()
                st.metric("Judged Faithful", f"{j2_mean:.1%}")

            with stat_cols[2]:
                st.metric("Rater A-B Agreement", f"{df['rater_agreement'].mean():.2%}")
            with stat_cols[3]:
                if kappa_calculated:
                    st.metric(
                        "Cohen's κ (inter-rater faithfulness)", f"{cohens_kappa:.4f}"
                    )
                else:
                    st.metric("Cohen's κ (inter-rater faithfulness)", "n/a")

            with stat_cols[4]:
                if len(df) > 0:
                    hallucination_rate = df["is_hallucination_failure"].mean()
                else:
                    hallucination_rate = 0.0

                st.metric("Hallucination Rate", f"{hallucination_rate:.1%}")

            st.markdown("#### Hallucination Accumulation by Depth")

            if "is_impossible" in df.columns:
                imp_df = df[df["is_impossible"] == True]
            else:
                df.iloc[0:0]

            try:
                if imp_df.empty:
                    st.info("No unanswerable (is_impossible) items in this sample")
                else:
                    max_hops = int(imp_df["iterations_completed"].max())
                    fab_rows = []
                    for hop in range(max_hops):
                        vals = []
                        for h in imp_df["fabrication_history"]:
                            if isinstance(h, list) and len(h) > hop:
                                vals.append(h[hop])

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

            except Exception as e:
                st.warning(f"Error: {e}")
            st.write("---")

            st.subheader("Agent History")

            selected_sample_id = st.selectbox(
                "Select Sample ID:", options=df["sample_id"].tolist()
            )

            if selected_sample_id:
                sample_data = df[df["sample_id"] == selected_sample_id].iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Topic:** {sample_data['topic']}")
                with c2:
                    if sample_data["rater_a_faithful"] == 1:
                        rat_a_lbl = "Faithful"
                    else:
                        rat_a_lbl = "Hallucinated"
                    st.markdown(f"**Rater A faithfulness:** {rat_a_lbl}")
                with c3:
                    if sample_data["rater_b_faithful"] == 1:
                        rat_b_lbl = "Faithful"
                    else:
                        rat_b_lbl = "Hallucinated"
                    st.markdown(f"**Rater B faithfulness:** {rat_b_lbl}")

                st.markdown("### Benchmark Reference Data")
                st.info(
                    f"Question asked: {sample_data['question']}\n\n Groundtruth answer: {sample_data['ground_truth']}\n\n All valid answers: {sample_data['all_valid_answers']}"
                )

                with st.expander("Original context", expanded=False):
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

                ground_truth_answer = sample_data.get("ground_truth", "N/A")
                if isinstance(
                    ground_truth_answer, str
                ) and ground_truth_answer.startswith("["):
                    try:
                        ground_truth_answer = ast.literal_eval(ground_truth_answer)
                    except Exception:
                        pass

                qa_labels = {0.0: "Preserved", 0.5: "Partial Loss", 1.0: "Lost"}

                for i, text in enumerate(text_history):
                    depth = i + 1
                    if i < len(semantic_history):
                        drift = semantic_history[i]
                    else:
                        drift = 0.0

                    similarity = 1.0 - drift
                    if i < len(qa_history):
                        qa_score = qa_history[i]
                    else:
                        qa_score = 0.0
                    qa_label = qa_labels.get(qa_score, f"{qa_score:.1f}")

                    is_expanded = depth == len(text_history)
                    expander_title = (
                        f"Hop {depth} Output — Semantic Similarity: {similarity:.1%} | "
                        f"Semantic Drift: {drift:.1%} | Answer Recoverability: {qa_label}"
                    )
                    if depth == len(text_history):
                        expander_title = "FINAL OUTPUT - " + expander_title

                    with st.expander(expander_title, expanded=is_expanded):
                        st.write(text)
                        if i < len(answers_history):
                            reader_ans = answers_history[i]
                        else:
                            reader_ans = "N/A"

                        if sample_data["is_impossible"]:
                            cor_ans = "N/A (Unanswerable)"
                            if "CANNOT_ANSWER" in str(reader_ans).upper():
                                verdict = "Correctly refused"
                            else:
                                verdict = "Fabricated"

                        else:
                            cor_ans = sample_data.get("ground_truth", "N/A")
                            verdict = "N/A (Answerable)"
                        st.caption(
                            f"Reader Answer: {reader_ans} | Correct Answer: {cor_ans} | Verdict: {verdict}"
                        )

            st.write("---")
            st.subheader("Export Results")
            if "run_id" in df.columns and len(df):
                run_tag = df["run_id"].iloc[0]
            else:
                run_tag = "run"

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Pipeline Results (CSV)",
                data=csv,
                file_name=(
                    f"drift_d{target_depth}_T{exp_temp}_seed{int(random_seed)}_{run_tag}.csv"
                ),
                mime="text/csv",
            )

            with st.expander("Run manifest"):
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
                present = []
                for c in manifest_cols:
                    if c in df.columns:
                        present.append(c)

                f_row = df[present].iloc[0]
                manifest_tab = f_row.rename("value").to_frame()
                manifest_tab = manifest_tab.astype(str)
                st.table(manifest_tab)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        import traceback

        st.error(traceback.format_exc())

else:
    st.info("upload the SQuAD v2 dataset")
