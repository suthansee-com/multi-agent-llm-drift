import streamlit as st
import re
import operator
from typing import TypedDict, List, Annotated
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
        nltk.data.find("tokenizers/punkt_tab")
        nltk.data.find("taggers/averaged_perceptron_tagger")
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("wordnet", quiet=True)


init_nltk()
lemmatizer = WordNetLemmatizer()


class AgentState(TypedDict):
    """A dictionary representing the state of an agent."""

    original_context: str
    current_text: str
    history: Annotated[List[str], operator.add]
    current_depth: int
    target_depth: int
    temperature: float
    token_preservation_scores: Annotated[List[float], operator.add]
    is_impossible: bool
    question: str
    ground_truth_answer: str


def calculate_core_token_preservation(original: str, current: str) -> float:
    """Calculates the preservation score of core tokens between the original and current text."""
    original_tokens = extract_content_tokens(original)
    current_tokens = extract_content_tokens(current)
    if not original_tokens:  
        return 0.0
    overlap = original_tokens.intersection(current_tokens)
    # union = original_tokens.union(current_tokens)
    return float(
        len(overlap) / len(original_tokens)
    ) 


def extract_content_tokens(text: str) -> set:
    """Extracts content tokens from the given text"""
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


def create_pipeline():
    """Creating a pipeline."""

    def rewrite_node(state: AgentState):
        llm = ChatOllama(model="llama3:8b", temperature=state["temperature"])
        system_prompt = SystemMessage(
            content=(
                "You are a strict text-processing relay agent. Rewrite the provided text "
                "completely, in your own words using alternative vocabulary and phrasing structures. "
                "Do NOT include any introductory expressions, conversational pleasantries, notes, or explanations. "
                "Output ONLY the raw processed text."
            )
        )
        human_prompt = HumanMessage(
            content=f"Text to rewrite:\n{state['current_text']}"
        )
        response = llm.invoke([system_prompt, human_prompt])
        raw_text = response.content.strip()
        rewritten_text = clean_llm_artifacts(raw_text)

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
def load_ollama_judge():
    return ChatOllama(model="llama3:8b", temperature=0.1)


def LLM_Evaluator_Judge_1(
    original_text: str, final_text: str, is_impossible: bool = False
) -> int:
    if is_impossible and "UNANSWERABLE" in final_text.upper():
        return 1

    judge_llm = load_ollama_judge()

    system_prompt = SystemMessage(
        content=(
            "You are a rigorous factual accuracy evaluator. "
            "Compare the 'Generated Text' against the 'Original Context'. "
            "Determine if all factual claims in the generated text are supported by the original context. "
            "Respond with ONLY 'ACCURATE' or 'HALLUCINATED'."
        )
    )
    human_prompt = HumanMessage(content=f"""
            Original Context (Ground Truth):
            {original_text}
            Generated Text (To Evaluate):
            {final_text}    
            Is the generated text factually accurate based on the original context?
            Respond with ONLY: ACCURATE or HALLUCINATED
        """)
    try:
        response = judge_llm.invoke([system_prompt, human_prompt])
        verdict = response.content.strip().upper()
        return 1 if verdict == "ACCURATE" else 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0


def LLM_Evaluator_Judge_2(
    original_text: str, final_text: str, is_impossible: bool = False
) -> int:
    if is_impossible and "UNANSWERABLE" in final_text.upper():
        return 1

    judge_llm = load_ollama_judge()

    system_prompt = SystemMessage(
        content=(
            "You are a semantic fidelity evaluator. "
            "Review if the 'Generated Text' preserves the key entities, relationships, and facts "
            "from the 'Original Context'. Check for: (1) Missing key facts, "
            "(2) Added false information, (3) Distorted relationships. "
            "Respond with ONLY 'PRESERVES_FIDELITY' or 'LOSES_FIDELITY'."
        )
    )
    human_prompt = HumanMessage(content=f"""
            Original Context (Ground Truth):
            {original_text}
            Generated Text (To Evaluate):
            {final_text}    
            Does the generated text preserve the semantic fidelity of the original?
            Respond with ONLY: PRESERVES_FIDELITY or LOSES_FIDELITY
        """)
    try:
        response = judge_llm.invoke([system_prompt, human_prompt])
        verdict = response.content.strip().upper()
        return 1 if verdict == "PRESERVES_FIDELITY" else 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_llm_artifacts(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\n*Note:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"\n*I (?:made|have made|tried to).*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\n*Here (?:are|is) (?:some|the) (?:changes|updates).*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    preambles = [
        r"^Here[\'s]* (?:is )?(?:my |the )?(?:rewritten|cleaned|corrected|revised|summarized) (?:version|text)[:\\.]?\s*",
        r"^Here[\'s]* (?:the )?(?:rewritten|cleaned|corrected|revised) (?:text|version)[:\\.]?\s*",
        r"^The (?:rewritten|cleaned|corrected|revised) text[:\\.]?\s*",
        r"^(?:Rewritten|Cleaned|Corrected|Revised) (?:version|text)[:\\.]?\s*",
        r"^Below (?:is )?(?:my )?(?:rewritten|cleaned|corrected|revised)[:\\.]?\s*",
        r"^I[\'ve]* (?:rewritten|cleaned|corrected|revised)[:\\.]?\s*",
    ]

    for pattern in preambles:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()
