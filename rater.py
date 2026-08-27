from langchain_ollama import ChatOllama
from configuration import RATER_MODEL_A, RATER_MODEL_B, faithfulness_rubric
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List
import streamlit as st


@st.cache_resource
def load_rater_A(seed: int):
    return ChatOllama(model=RATER_MODEL_A, temperature=0.0, seed=seed)


@st.cache_resource
def load_rater_B(seed: int):
    return ChatOllama(model=RATER_MODEL_B, temperature=0.0, seed=seed)


def rate_faithfulness_A(
    original_text: str, final_text: str, is_impossible: bool = False, seed: int = 42
) -> int:

    rater_llm = load_rater_A(seed)

    system_prompt = SystemMessage(
        content=(
            f"You are a rigorous, literal-minded fact checker.\n\n{faithfulness_rubric}"
        )
    )
    human_prompt = HumanMessage(content=f"""
            Original Context (Ground Truth):
            {original_text}
            Generated Text (To Evaluate):
            {final_text}    
            Is the rewritten passage FAITHFUL to the original source?
Respond with ONLY: FAITHFUL or HALLUCINATED
        """)
    try:
        response = rater_llm.invoke([system_prompt, human_prompt])
        verdict = response.content.strip().upper()
        if "FAITHFUL" in verdict and "HALLUCINATED" not in verdict:
            return 1
        return 0

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0


def rate_faithfulness_B(
    original_text: str, final_text: str, is_impossible: bool = False, seed: int = 42
) -> int:

    rater_llm = load_rater_B(seed)

    system_prompt = SystemMessage(
        content=(
            f"You are a careful, skeptical editor checking for unsupported claims.\n\n{faithfulness_rubric}"
        )
    )
    human_prompt = HumanMessage(
        content=f"""
            Original Context (Ground Truth):
            {original_text}
            Generated Text (To Evaluate):
            {final_text}    
            Is the rewritten passage FAITHFUL to the original source? Respond with ONLY: FAITHFUL or HALLUCINATED"""
    )
    try:
        response = rater_llm.invoke([system_prompt, human_prompt])
        verdict = response.content.strip().upper()
        if "FAITHFUL" in verdict and "HALLUCINATED" not in verdict:
            return 1
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0


def rate_faithfulness_at_checkpoints(
    original_context: str,
    history: List[str],
    is_impossible: bool = False,
    seed: int = 42,
) -> dict:
    depth = len(history)
    results = {}
    for hop in select_checkpoints(depth):
        text_at_hop = history[hop - 1]
        a = rate_faithfulness_A(original_context, text_at_hop, is_impossible, seed)
        b = rate_faithfulness_B(original_context, text_at_hop, is_impossible, seed)
        results[hop] = {
            "rater_a": a,
            "rater_b": b,
            "both_faithful": int(a == 1 and b == 1),
        }
    return results


def select_checkpoints(depth):
    if depth <= 1:
        return [1]

    midpoint = (depth + 1) // 2
    checkpoints = []

    for i in range(1, depth + 1):
        if i == 1 or i == depth or (depth > 3 and i == midpoint):
            checkpoints.append(i)

    return checkpoints
