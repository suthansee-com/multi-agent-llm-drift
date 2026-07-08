import random
import streamlit as st
import json
import pandas as pd
import numpy as np
import re
import operator
import math
from sklearn.metrics import cohen_kappa_score
from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
from langgraph.graph import StateGraph, END

from pipeline import AgentState, calculate_core_token_preservation, extract_content_tokens, create_pipeline, init_nltk, lemmatizer, LLM_Evaluator_Judge_1, LLM_Evaluator_Judge_2, LLM_Evaluator_Judge_3_QA_Answerable, measure_semantic_drift, extract_agent_understanding


@st.cache_resource
def get_pipeline_app():
    return create_pipeline()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

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

        total_records = len(all_data)

        # total_records = len(all_data)
        # unique_contexts = {}
        # for record in records:
        #     title = record.get("title", "Unknown")
        #     for para in record.get("paragraphs", []):
        #         context_text = para.get("context", "").strip()
        #         if context_text and context_text not in unique_contexts:
        #             # Extract the first question ID just as a reference handle
        #             first_qa_id = para.get("qas", [{'id': '0'}])[0].get("id", "")
        #             unique_contexts[context_text] = {
        #                 "source_text_id": f"{title}_{first_qa_id}",
        #                 "topic": title,
        #                 "context": context_text,
        #             }

        # all_data = list(unique_contexts.values())
        # total_records = len(all_data)
        st.success(
            f"Dataset loaded Successfully! Total available records : {total_records}"
        )
        st.write("---")

        st.subheader("Experimental Configuration")
        col1, col2 = st.columns([1, 2])

        with col1:
            sample_size = st.number_input(
                "Sample Size (N):",
                min_value=1,
                max_value=min(200, total_records),
                value=1,
                step=1,
            )
            target_depth = st.selectbox("Pipeline Depth (n):", [1, 5, 10], index=0)
            exp_temp = st.selectbox("Generation Temperature (T):", [0.1, 0.5, 0.9], index=1)
            random_seed = st.number_input(
                "Random Seed:", min_value=1, value=42
            )
            generate_btn = st.button("Generate Experiment", type="primary")

        if generate_btn:
            random.seed(random_seed)
            selected_sample = random.sample(all_data, min(sample_size, len(all_data)))
            valid_sample = [row for row in selected_sample if row.get("context")]

            st.write("---")
            st.header("Pipeline Execution")
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []
            judge1_verdicts = []
            judge2_verdicts = []
            judge3_verdicts = []
            
            for idx, row in enumerate(valid_sample):
                status_text.text(f"Processing item {idx + 1} of {len(valid_sample)}...")

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
                    "token_preservation_scores": [],
                }

                progress_bar.progress((idx + 1) / len(valid_sample))

                final_result = pipeline_app.invoke(initial_state)
                # scores = final_result.get("token_preservation_scores", [])
                semantic_scores = final_result.get("semantic_understanding_scores", []) # ADD THIS LINE

                status_text.text(f"Processing {idx + 1}/{len(valid_sample)} - Evaluating final hop output with dual judges...")

                judge1_verdict = LLM_Evaluator_Judge_1(
                    row["context"], final_result.get("current_text", ""), row["is_impossible"]
                )
                judge2_verdict = LLM_Evaluator_Judge_2(
                    row["context"], final_result.get("current_text", ""), row["is_impossible"]
                )
                judge3_verdict = LLM_Evaluator_Judge_3_QA_Answerable(
                    rewritten_context=final_result.get("current_text", ""), question=row["question"], expected_answer=row["ground_truth_answer"], is_impossible=row["is_impossible"]
                )

                judge1_verdicts.append(judge1_verdict)
                judge2_verdicts.append(judge2_verdict)
                judge3_verdicts.append(judge3_verdict)

                # avg_score = float(np.mean(scores)) if scores else 0.0
                # final_score = float(scores[-1]) if scores else 0.0

                # status_text.text(f"Processing {idx + 1}/{len(valid_sample)} - Computing embedding-based drift...")
                
                # try:
                #     embedding_original = embedder.encode(row["context"])
                #     embedding_final = embedder.encode(final_result.get("current_text", ""))
                #     cosine_sim = cosine_similarity([embedding_original], [embedding_final])[0][0]
                #     embedding_drift = float(1.0 - cosine_sim)
                #     embedding_similarity = float(cosine_sim)
                # except:
                #     embedding_drift = 0.0
                #     embedding_similarity = 0.0

                # drift_threshold = 0.20 
                # avg_drift_val = 1.0 - avg_score

                # if judge1_verdict == 1 and judge2_verdict == 1:
                #     if avg_drift_val >= drift_threshold:
                #         failure_mode = "Linguistic Rephrasing"
                #     else:
                #         failure_mode = "Stable / Conserved"
                # else:
                #     if row["is_impossible"]:
                #         if "UNANSWERABLE" in final_result["current_text"].upper():
                #             failure_mode = "Correct Refusal (System Success)"
                #         else:
                #             failure_mode = "Belief State Failure (Hallucinated Fact)"
                #     else:
                #         failure_mode = "Belief State Failure (Loss of Fidelity)"
                final_semantic_drift = semantic_scores[-1] if semantic_scores else 0.0

                if judge1_verdict == 1 and judge2_verdict == 1:
                    if final_semantic_drift > 0:
                        failure_mode = "Semantic Shift"
                    else:
                        failure_mode = "Stable / Conserved"
                else:
                    if row["is_impossible"]:
                        if "UNANSWERABLE" in final_result["current_text"].upper():
                            failure_mode = "Correct Refusal (System Success)"
                        else:
                            failure_mode = "Belief State Failure (Hallucinated Fact)"
                    else:
                        failure_mode = "Belief State Failure (Loss of Fidelity)"
                

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
                        # "scores_history": scores,
                        "semantic_scores_history": semantic_scores,
                        "pipeline_depth": target_depth,
                        "temperature": exp_temp,
                        "iterations_completed": final_result["current_depth"],
                        # "avg_token_preservation": avg_score,
                        # "final_token_preservation": final_score,
                        # "avg_drift": 1.0 - avg_score,
                        # "final_drift": 1.0 - final_score,
                        # "embedding_drift": embedding_drift,
                        # "embedding_similarity": embedding_similarity,
                        "judge1_verdict": judge1_verdict,
                        "judge2_verdict": judge2_verdict,
                        "judge3_qa_answerable": judge3_verdict,
                        "diagnosed_cause": failure_mode,
                        "is_belief_failure": 1 if failure_mode.startswith("Belief State Failure") else 0,
                        # "is_rephrased": 1 if failure_mode == "Linguistic Rephrasing" else 0,
                        "judge_agreement": 1 if judge1_verdict == judge2_verdict else 0,
                        # "preservation_history": str(scores),
                    }
                )

                progress_bar.progress(min(1.0, (idx + 1) / len(valid_sample)))

            status_text.empty()
            progress_bar.empty()

            st.session_state["results_df"] = pd.DataFrame(results_list)
            st.success("Pipeline execution complete!")
            st.rerun()

        if "results_df" in st.session_state:
            df = st.session_state["results_df"]

            judge1_verdicts = df["judge1_verdict"].tolist()
            judge2_verdicts = df["judge2_verdict"].tolist()
            judge3_verdicts = df["judge3_qa_answerable"].tolist()

            if len(judge1_verdicts) >= 2:
                #cohens_kappa = cohen_kappa_score(judge1_verdicts, judge2_verdicts)
                cohens_kappa_1_2 = cohen_kappa_score(judge1_verdicts, judge2_verdicts)
                cohens_kappa_1_3 = cohen_kappa_score(judge1_verdicts, judge3_verdicts)

                cohens_kappa = (cohens_kappa_1_2 + cohens_kappa_1_3) / 2
            else:
                cohens_kappa = 0.0

            agreement_list = [1 if v1 == v2 else 0 for v1, v2 in zip(judge1_verdicts, judge2_verdicts)]
            avg_agreement_pct = (sum(agreement_list) / len(agreement_list)) * 100 if agreement_list else 0.0

            if math.isnan(cohens_kappa) or np.isnan(cohens_kappa):
                cohens_kappa = 1.0 if avg_agreement_pct == 100.0 else 0.0

            st.header("4. Pipeline Results")
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)

            # with col_m1:
            #     st.metric("Samples Processed", len(df))
            # with col_m2:
            #     st.metric(
            #         "Pipeline Depth",
            #         int(df["pipeline_depth"].iloc[0]) if len(df) > 0 else 0,
            #     )
            # with col_m3:
            #     st.metric(
            #         "Temperature",
            #         float(df["temperature"].iloc[0]) if len(df) > 0 else 0,
            #     )
            # with col_m4:
            #     st.metric(
            #         "Avg Token Preservation",
            #         f"{df['avg_token_preservation'].mean():.2%}",
            #     )
            # with col_m5:
            #     st.metric(
            #         "Final Preservation", f"{df['final_token_preservation'].mean():.2%}"
            #     )
            # with col_m6:
            #     st.metric("Avg Drift", f"{df['avg_drift'].mean():.2%}")
           
            # st.write("---")

            # col_m7, col_m8, col_m9, col_m10 = st.columns(4)

            # with col_m7:
            #     st.metric("Embedding Drift", f"{df['embedding_drift'].mean():.2%}")
            # with col_m8:
            #     st.metric("Judge Agreement", f"{df['judge_agreement'].mean():.2%}")
            # with col_m9:
            #     st.metric("Cohen's Kappa", f"{cohens_kappa:.4f}")  
            # with col_m10:
            #     belief_failure_rate = df["is_belief_failure"].mean() if len(df) > 0 else 0.0
            #     st.metric("Belief Failure Rate", f"{belief_failure_rate:.1%}")
            # # with col_m11:
            # #     judge3_agreement = df["judge3_qa_answerable"].mean() if len(df) > 0 else 0.0
            # #     st.metric("Judge 3 (Q&A Answerable)", f"{judge3_agreement:.1%}")
            st.header("4. Pipeline Results")
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric("Samples Processed", len(df))
            with col_m2:
                st.metric("Pipeline Depth", int(df["pipeline_depth"].iloc[0]) if len(df) > 0 else 0)
            with col_m3:
                st.metric("Temperature", float(df["temperature"].iloc[0]) if len(df) > 0 else 0)
           
            st.write("---")

            col_m8, col_m9, col_m10 = st.columns(3)

            with col_m8:
                st.metric("Judge Agreement", f"{df['judge_agreement'].mean():.2%}")
            with col_m9:
                st.metric("Cohen's Kappa", f"{cohens_kappa:.4f}")  
            with col_m10:
                belief_failure_rate = df["is_belief_failure"].mean() if len(df) > 0 else 0.0
                st.metric("Belief Failure Rate", f"{belief_failure_rate:.1%}")

            

            st.markdown("### Statistical Summary")
            
            stat_cols = st.columns(2)
            
            with stat_cols[0]:
                st.write("**Judge 1 (Factual)**")
                j1_mean = df["judge1_verdict"].mean()
                st.metric("Accuracy", f"{j1_mean:.1%}")
            
            with stat_cols[1]:
                st.write("**Judge 3 (Q&A)**")
                j3_mean = df["judge3_qa_answerable"].mean()
                st.metric("Answerable", f"{j3_mean:.1%}")
            
            st.markdown("---")

            st.write("---")
            st.subheader("Agent History & Audit Viewer")
            st.write("Select a sample below to inspect exact propagation hops and ground-truth alignment:")

            selected_sample_id = st.selectbox(
                "Select Sample ID:", options=df["sample_id"].tolist()
            )

            if selected_sample_id:
                sample_data = df[df["sample_id"] == selected_sample_id].iloc[0]
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Topic:** {sample_data['topic']}")
                with c2:
                    st.markdown(f"**Judge 1 (Factual Audit):** {'Accurate' if sample_data['judge1_verdict'] == 1 else 'Hallucinated'}")
                with c3:
                    st.markdown(f"**Judge 2 (Fidelity Audit):** {'Preserved' if sample_data['judge2_verdict'] == 1 else 'Lost'}")
                
                st.markdown("### Benchmark Reference Data")
                st.info(f"**Question Asked:** {sample_data['question']}\n\n**Gold Standard Answer:** {sample_data['ground_truth']}\n\n**All Valid Answers:** {sample_data['all_valid_answers']}")
                
                with st.expander("Background SQuAD Context (Input at Hop 0)", expanded=False):
                    st.write(sample_data["original_text_full"])
                    
                st.markdown("### Agent Chain Propagation History")
                text_history = sample_data["text_history"]
                semantic_history = sample_data["semantic_scores_history"]
                
                if isinstance(semantic_history, str):
                    import ast
                    semantic_history = ast.literal_eval(semantic_history)
                
                for i, text in enumerate(text_history):
                    depth = i + 1
                    score = semantic_history[i] if i < len(semantic_history) else 0.0
                    drift = 1.0 - score
                    
                    is_expanded = (depth == len(text_history))
                    expander_title = f"Hop {depth} Output — Semantic QA Alignment: {score:.1%} | Drift: {drift:.1%}"
                    if depth == len(text_history):
                        expander_title = "FINAL OUTPUT — " + expander_title
                        
                    with st.expander(expander_title, expanded=is_expanded):
                        st.write(text)

                st.subheader("Information Degradation Curve (Semantic Drift)")
                
                semantic_list = sample_data["semantic_scores_history"]
                
                if isinstance(semantic_list, str):
                    import ast
                    semantic_list = ast.literal_eval(semantic_list)
                
                chart_data = pd.DataFrame({
                    "Hop": range(1, len(semantic_list) + 1),
                    "Semantic QA Drift (%)": [s * 100 for s in semantic_list]
                }).set_index("Hop")
                
                st.line_chart(chart_data, color=["#ffc107"])


            st.markdown("### Experiment Configuration and Metadata")
            
            metadata_cols = st.columns(7)
            with metadata_cols[0]:
                st.metric("Pipeline Depth", int(target_depth))
            with metadata_cols[1]:
                st.metric("Temperature", exp_temp)
            with metadata_cols[2]:
                st.metric("Sample Size (N)", len(df))
            with metadata_cols[3]:
                st.metric("Random Seed", random_seed)
            with metadata_cols[4]:
                st.metric("Model", "gemma2:2b")
            with metadata_cols[5]:
                st.metric("Dataset", "SQuAD v2.0")
            with metadata_cols[6]:
                import datetime
                st.metric("Run Date", datetime.datetime.now().strftime("%Y-%m-%d"))
            
            st.markdown("---")

            import plotly.graph_objects as go
            
            st.markdown("### Judge Agreement Analysis")
            col_j_agree, col_failure = st.columns(2)
            
            with col_j_agree:
                st.write("**Judge Verdict Agreement**")
                # Count agreement patterns
                all_agree = (df["judge1_verdict"] == df["judge2_verdict"]) & \
                           (df["judge2_verdict"] == df["judge3_qa_answerable"])
                two_agree = ~all_agree & \
                           ((df["judge1_verdict"] == df["judge2_verdict"]) | \
                            (df["judge1_verdict"] == df["judge3_qa_answerable"]) | \
                            (df["judge2_verdict"] == df["judge3_qa_answerable"]))
                disagree = ~all_agree & ~two_agree
                
                agreement_counts = [
                    all_agree.sum(),
                    two_agree.sum(),
                    disagree.sum()
                ]
                agreement_labels = ["All 3 Agree", "2/3 Agree", "Disagreement"]
                
                fig = go.Figure(data=[go.Pie(
                    labels=agreement_labels,
                    values=agreement_counts,
                    marker=dict(colors=["#28a745", "#ffc107", "#dc3545"])
                )])
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_failure:
                st.write("**Failure Mode Distribution**")
                failure_counts = df["diagnosed_cause"].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=failure_counts.index,
                    values=failure_counts.values,
                    marker=dict(colors=["#0d6efd", "#198754", "#fd7e14", "#6c757d"])
                )])
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Judge Verdict Heatmap (Sample × Judge Agreement)")
            
            heatmap_df = df[["topic", "judge1_verdict", "judge2_verdict", 
                             "judge3_qa_answerable", "diagnosed_cause"]].copy()
            heatmap_df["All_Agree"] = (
                (heatmap_df["judge1_verdict"] == heatmap_df["judge2_verdict"]) & 
                (heatmap_df["judge2_verdict"] == heatmap_df["judge3_qa_answerable"])
            ).astype(int)
            heatmap_df = heatmap_df.rename(columns={
                "judge1_verdict": "Judge 1 (Factual)",
                "judge2_verdict": "Judge 2 (Fidelity)",
                "judge3_qa_answerable": "Judge 3 (Q&amp;A)",
                "All_Agree": "All Agree",
                "diagnosed_cause": "Failure Mode"
            })
            
            st.dataframe(
                heatmap_df.style.map(lambda x: 'background-color: #28a745; color: white' 
                                     if x == 1 else 'background-color: #dc3545; color: white' 
                                     if x == 0 else ''),
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            st.subheader("Detailed Results")
            display_cols = [
                # "sample_id",
                # "topic",
                # "avg_token_preservation",
                # "final_token_preservation",
                # "original_text",
                # "final_text",
                "topic",
                "question",
                "ground_truth",
                "final_text",
                # "avg_drift",
                # "embedding_drift",
                "judge1_verdict",       
                "judge2_verdict",        
                "judge3_qa_answerable",  
                "diagnosed_cause",
            ]
            st.dataframe(df[display_cols], use_container_width=True, height=400)

            st.write("---")
            st.subheader("Export Results")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Pipeline Results (CSV)",
                data=csv,
                file_name=f"pipeline_results_d{target_depth}_t{exp_temp}.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        import traceback

        st.error(traceback.format_exc())

else:
    st.info("Please upload the SQuAD dev-v2 dataset (JSON) to begin")
