import streamlit as st
import re
import operator
import string
from collections import Counter
from typing import TypedDict, List, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

SEMANTIC_SHIFT_THRESHOLD = 0.15

AGENT_MODEL = "gemma2:2b"
READER_MODEL = "gemma2:2b"
RATER_MODEL_A = "llama3:8b"
RATER_MODEL_B = "mistral:7b"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"


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


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBEDDER_MODEL)


class AgentState(TypedDict):
    """A dictionary representing the state of an agent."""
    original_context: str
    current_text: str
    question: str
    ground_truth_answer: str
    history: Annotated[List[str], operator.add]
    current_depth: int
    target_depth: int
    temperature: float
    lexical_preservation_scores: Annotated[List[float], operator.add]
    is_impossible: bool
    semantic_drift_scores: Annotated[List[float], operator.add]
    global_drift_scores: Annotated[List[float], operator.add]
    qa_recoverability_scores: Annotated[List[float], operator.add]
    em_recoverability_scores: Annotated[List[float], operator.add]
    agent_answers: Annotated[List[str], operator.add]
    fabrication_flags: Annotated[List[int], operator.add]
    seed: int
    all_valid_answers: List[str]



def calculate_core_token_preservation(original: str, current: str) -> float:
    """Calculates the preservation score of core tokens between the original and current text."""
    original_tokens = extract_content_tokens(original)
    current_tokens = extract_content_tokens(current)
    if not original_tokens:
        return 0.0
    overlap = original_tokens.intersection(current_tokens)
    # union = original_tokens.union(current_tokens)
    return float(len(overlap) / len(original_tokens))


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


def extract_agent_understanding(context: str, question: str, seed: int = 42) -> dict:
    """
    Extract what the agent understands from the context.
    This is the core of semantic drift measurement.
    
    Returns:
        Dictionary with key_facts, understanding, and answer
    """
    llm = ChatOllama(model=READER_MODEL, temperature=0.0, seed=seed)
    
    system_prompt = SystemMessage(
        content=(
            "You are a careful reader. Analyze the provided text and extract key information.\n"
            "Be precise and concise in your responses."
        )
    )
    
    human_prompt = HumanMessage(
        content=f"""Text: {context}
 
Question: {question}
 
Please provide:
1. KEY_FACTS: List the key facts from the text needed to answer the question
2. ANSWER: Answer the question directly from the text, or say CANNOT_ANSWER
 
Format your response exactly as:
KEY_FACTS: [your facts]
ANSWER: [your answer]"""
    )
    
    try:
        response = llm.invoke([system_prompt, human_prompt])
        response_text = response.content.strip()
        
        understanding = {
            "full_response": response_text,
            "key_facts": "",
            "answer": ""
        }
        
        if "KEY_FACTS:" in response_text:
            facts_start = response_text.find("KEY_FACTS:") + len("KEY_FACTS:")
            facts_end = response_text.find("ANSWER:") if "ANSWER:" in response_text else len(response_text)
            understanding["key_facts"] = response_text[facts_start:facts_end].strip()
        
        if "ANSWER:" in response_text:
            answer_start = response_text.find("ANSWER:") + len("ANSWER:")
            understanding["answer"] = response_text[answer_start:].strip()
        
        return understanding
    except Exception as e:
        print(f"Error extracting understanding: {e}")
        return {
            "full_response": "",
            "key_facts": "",
            "answer": ""
        }

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD official token-level F1"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

    
def squad_em(prediction: str, ground_truth: str) -> float:
    """SQuAD official exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def measure_em_recoverability(
    current_understanding: dict,
    expected_answers,
    is_impossible: bool = False,
) -> float:
    current_answer = current_understanding.get("answer", "").lower().strip()

    if is_impossible:
        return 0.0 if "cannot_answer" in current_answer else 1.0

    if "cannot_answer" in current_answer:
        return 1.0

    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]

    golds = [g for g in (expected_answers or []) if isinstance(g, str) and g.strip()]
    if not golds:
        return 1.0

    best_em = max(squad_em(current_answer, g) for g in golds)
    return 1.0 - best_em


def measure_qa_recoverability(
    current_understanding: dict,
    expected_answers,
    is_impossible: bool = False,
) -> float:
    current_answer = current_understanding.get("answer", "").lower().strip()

    if is_impossible:
        return 0.0 if "cannot_answer" in current_answer else 1.0

    if "cannot_answer" in current_answer:
        return 1.0

    if isinstance(expected_answers, str):
        expected_answers = [expected_answers]

    golds = [g for g in (expected_answers or []) if isinstance(g, str) and g.strip()]
    if not golds:
        return 1.0

    best_f1 = max(squad_f1(current_answer, g) for g in golds)
    return 1.0 - best_f1


def calculate_semantic_drift(previous_text: str, current_text: str) -> float:
    try:
        embedder = get_embedder()
        embeddings = embedder.encode([previous_text, current_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(min(1.0, max(0.0, 1.0 - similarity)))
    except Exception as e:
        print(f"Semantic drift error: {e}")
        return 1.0


def create_pipeline():
    """Creating a pipeline."""

    def rewrite_node(state: AgentState):
        llm = ChatOllama(model=AGENT_MODEL, temperature=state["temperature"], seed=state["seed"])
        system_prompt = SystemMessage(
            content=(
                "You are a strict text-processing relay agent. Rewrite the provided text "
                "completely, in your own words using alternative vocabulary and phrasing structures. "
                "Do NOT include any introductory expressions, conversational pleasantries, notes, or explanations. "
                "Output ONLY the raw processed text."
            )
        )
        human_prompt = HumanMessage(content=f"""Text to rewrite:
            {state['current_text']}

            Rewrite the text completely in your own words. Use alternative vocabulary and phrasing structures.
            Do NOT include any introductory expressions, conversational pleasantries, notes, or explanations.
            Output ONLY the raw rewritten text.""")

        response = llm.invoke([system_prompt, human_prompt])
        raw_text = response.content.strip()
        rewritten_text = clean_llm_artifacts(raw_text)

        preservation_score = calculate_core_token_preservation(
            state["original_context"], rewritten_text
        )

        current_understanding = extract_agent_understanding(
            rewritten_text,
            state["question"],
            state["seed"]
        )
        agent_answer = current_understanding.get("answer", "CANNOT_ANSWER")

        semantic_drift_metric = calculate_semantic_drift(
            state["current_text"], rewritten_text
        )


        global_drift_metric = calculate_semantic_drift(
            state["original_context"], rewritten_text
        )

        qa_recoverability = measure_qa_recoverability(
            current_understanding,
            state["all_valid_answers"],
            state["is_impossible"]
        )

        em_recoverability = measure_em_recoverability(
            current_understanding,
            state["all_valid_answers"],
            state["is_impossible"]
        )


        _ans = current_understanding.get("answer", "").lower().strip()
        fabricated = 1 if (state["is_impossible"] and "cannot_answer" not in _ans) else 0


        return {
            "current_text": rewritten_text,
            "history": [rewritten_text],
            "current_depth": state["current_depth"] + 1,
            "lexical_preservation_scores": [preservation_score],
            "semantic_drift_scores": [semantic_drift_metric],
            "global_drift_scores": [global_drift_metric],
            "qa_recoverability_scores": [qa_recoverability],
            "em_recoverability_scores": [em_recoverability],
            "agent_answers": [agent_answer],
            "fabrication_flags": [fabricated],
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
def load_rater_A(seed: int):
    return ChatOllama(model=RATER_MODEL_A, temperature=0.0, seed=seed)

@st.cache_resource
def load_rater_B(seed: int):
    return ChatOllama(model=RATER_MODEL_B, temperature=0.0, seed=seed)

faithfulness_rubric = (
    "Decide whether the REWRITTEN passage is FAITHFUL to the ORIGINAL source. "
    "It is FAITHFUL only if every factual claim it makes is supported by the "
    "original, with no invented facts, no dropped facts that change the meaning, "
    "and no distorted relationships between entities. If any claim is unsupported, "
    "invented, or contradicts the source, it is HALLUCINATED. "
    "Respond with ONLY one word: FAITHFUL or HALLUCINATED."
)

def rate_faithfulness_A(
    original_text: str, final_text: str, is_impossible: bool = False, seed: int = 42
) -> int:

    rater_llm = load_rater_A(seed)

    system_prompt = SystemMessage(
        content=(f"You are a rigorous, literal-minded fact checker.\n\n{faithfulness_rubric}")
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
        return 1 if ("FAITHFUL" in verdict and "HALLUCINATED" not in verdict) else 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0


def rate_faithfulness_B(
    original_text: str, final_text: str, is_impossible: bool = False, seed: int = 42
) -> int:

    rater_llm = load_rater_B(seed)

    system_prompt = SystemMessage(
        content=(f"You are a careful, skeptical editor checking for unsupported claims.\n\n{faithfulness_rubric}")
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
        return 1 if ("FAITHFUL" in verdict and "HALLUCINATED" not in verdict) else 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0



def select_checkpoints(depth: int) -> List[int]:
    if depth <= 1:
        return [1]
    if depth <= 3:
        return sorted({1, depth})
    return sorted({1, (depth + 1) // 2, depth})


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
            "both_faithful": 1 if (a == 1 and b == 1) else 0,
        }
    return results


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