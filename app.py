import random
import streamlit as st
import json
import pandas as pd
import numpy as np
import re
import operator
from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


@st.cache_resource
def init_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("wordnet", quiet=True)


init_nltk()

lemmatizer = WordNetLemmatizer()


class AgentState(TypedDict):
    original_context: str
    current_text: str
    history: Annotated[List[str], operator.add]
    current_depth: int
    target_depth: int
    temperature: float
    token_preservation_scores: Annotated[List[float], operator.add]


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN



def extract_content_tokens(text: str) -> set:
    try:
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)

        content_tokens = set()
        for word, pos in pos_tags:
            if any(pos.startswith(prefix) for prefix in ["NN", "VB", "JJ", "RB"]):
                clean_word = "".join(c for c in word if c.isalnum())
                if len(clean_word) > 2:
                    wn_pos = get_wordnet_pos(pos)
                    lemma = lemmatizer.lemmatize(clean_word, pos=wn_pos)
                    content_tokens.add(lemma)

        if not content_tokens:
            content_tokens = set([w for w in text.lower().split() if len(w) > 2])
        return content_tokens

    except Exception as e:
        print(f"Token extraction error: {e}")
        return set([w for w in text.lower().split() if len(w) > 2])


def calculate_core_token_preservation(original: str, current: str) -> float:
    original_tokens = extract_content_tokens(original)
    current_tokens = extract_content_tokens(current)

    if not original_tokens or not current_tokens:
        return 0.0

    overlap = original_tokens.intersection(current_tokens)
    union = original_tokens.union(current_tokens)
    return float(len(overlap) / len(union))


def create_pipeline():
    def rewrite_node(state: AgentState):
        llm = ChatOllama(model="llama3:8b", temperature=state["temperature"])

        system_prompt = SystemMessage(
            content=(
                "You are a strict text-processing relay agent. Rewrite the provided text "
                "completely in your own words using alternative vocabulary and phrasing structures. "
                "Do NOT include any introductory expressions, conversational pleasantries, notes, or explanations. "
                "Output ONLY the raw processed text."
            )
        )

        human_prompt = HumanMessage(
            content=f"Text to rewrite:\n{state['current_text']}"
        )

        response = llm.invoke([system_prompt, human_prompt])

        rewritten_text = clean_llm_artifacts(response.content)

        preservation_score = calculate_core_token_preservation(
            state["original_context"], rewritten_text
        )

        return {
            "current_text": rewritten_text,
            "history": [rewritten_text],
            "current_depth": state["current_depth"] + 1,
            "token_preservation_scores": [preservation_score],
        }

    def should_continue(state: AgentState):
        if state["current_depth"] >= state["target_depth"]:
            return "end"
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", rewrite_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "agent", "end": END}
    )
    return workflow.compile()


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
                for qa in para.get("qas", []):
                    all_data.append(
                        {
                            "source_text_id": f"{title}_{qa.get('id', '')}",
                            "topic": title,
                            "question": qa.get("question"),
                            "context": context_text,
                            "answers": [
                                a.get("text", "") for a in qa.get("answers", [])
                            ],
                        }
                    )

        total_records = len(all_data)
        st.success(
            f"Dataset loaded Successfully! Total available records : {total_records}"
        )
        st.write("---")

        st.subheader("Experimental Configuration")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Parameters")
            sample_size = st.number_input(
                "Sample Size (N):",
                min_value=1,
                max_value=min(200, total_records),
                value=30,
                step=1,
            )
            target_depth = st.selectbox("Pipeline Depth (n):", [1, 5, 10], index=0)
            exp_temp = st.selectbox(
                "Generation Temperature (T):", [0.1, 0.5, 0.9], index=1
            )
            random_seed = st.number_input(
                "Random Seed:", min_value=1, max_value=999, value=42
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

            for idx, row in enumerate(valid_sample):
                status_text.text(f"Processing item {idx + 1} of {len(valid_sample)}...")

                initial_state = {
                    "original_context": row["context"],
                    "current_text": row["context"],
                    "history": [],
                    "current_depth": 0,
                    "target_depth": target_depth,
                    "temperature": exp_temp,
                    "token_preservation_scores": [],
                }

                progress_bar.progress((idx + 1) / len(valid_sample))

                final_result = pipeline_app.invoke(initial_state)
                scores = final_result.get("token_preservation_scores", [])

                avg_score = float(np.mean(scores)) if scores else 0.0
                final_score = float(scores[-1]) if scores else 0.0

                results_list.append(
                    {
                        "sample_id": row["source_text_id"],
                        "topic": row["topic"],
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
                        "pipeline_depth": target_depth,
                        "temperature": exp_temp,
                        "iterations_completed": final_result["current_depth"],
                        "avg_token_preservation": avg_score,
                        "final_token_preservation": final_score,
                        "preservation_history": str(
                            scores
                        ),  
                    }
                )

            status_text.empty()
            progress_bar.empty()
            st.session_state["results_df"] = pd.DataFrame(results_list)
            st.success("Pipeline execution complete!")
            st.rerun()

        if "results_df" in st.session_state:
            df = st.session_state["results_df"]

            st.header("4. Pipeline Results")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

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
            with col_m4:
                st.metric(
                    "Avg Token Preservation",
                    f"{df['avg_token_preservation'].mean():.2%}",
                )
            with col_m5:
                st.metric(
                    "Final Preservation", f"{df['final_token_preservation'].mean():.2%}"
                )

            st.write("---")
            st.subheader("Detailed Results")
            display_cols = [
                "sample_id",
                "topic",
                "avg_token_preservation",
                "final_token_preservation",
                "original_text",
                "final_text",
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
