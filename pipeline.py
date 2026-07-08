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
    question: str
    ground_truth_answer: str
    history: Annotated[List[str], operator.add]
    current_depth: int
    target_depth: int
    temperature: float
    token_preservation_scores: Annotated[List[float], operator.add]
    is_impossible: bool
    semantic_understanding_scores: Annotated[List[int], operator.add]
    agent_answers: Annotated[List[str], operator.add]
    


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


def extract_agent_understanding(context: str, question: str) -> dict:
    """
    Extract what the agent understands from the context.
    This is the core of semantic drift measurement.
    
    Returns:
        Dictionary with key_facts, understanding, and answer
    """
    llm = ChatOllama(model="gemma2:2b", temperature=0.1)
    
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


def measure_semantic_drift(original_understanding: dict, current_understanding: dict, expected_answer: str) -> float:
    """
    Measure semantic drift by comparing:
    1. Does agent still provide an answer (not CANNOT_ANSWER)?
    2. Does the answer match expected answer?
    
    Returns:
        0.0 if understanding completely lost (same as original)
        1.0 if understanding fully degraded (cannot answer)
        0.5 if partial degradation (wrong answer but attempted)
    
    Note: Higher score = more drift (opposite of preservation)
    """
    original_answer = original_understanding.get("answer", "").lower().strip()
    current_answer = current_understanding.get("answer", "").lower().strip()
    expected = expected_answer.lower().strip()
    
    if "cannot_answer" in current_answer:
        return 1.0
    
    if "cannot_answer" not in original_answer:
        if expected in current_answer or any(word in current_answer for word in expected.split()):
            return 0.0 
        else:
            return 0.5  
    
    return 0.0  


def create_pipeline():
    """Creating a pipeline."""

    def rewrite_node(state: AgentState):
        llm = ChatOllama(model="gemma2:2b", temperature=state["temperature"])
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

        # preservation_score = calculate_core_token_preservation(
        #     state["original_context"], rewritten_text
        # )

        current_understanding = extract_agent_understanding(
                    rewritten_text, 
                    state["question"]
                )


        agent_answer = current_understanding.get("answer", "CANNOT_ANSWER")

        # if state["current_depth"] == 1:
        #     # First hop: extract understanding from original context
        #     original_understanding = extract_agent_understanding(
        #         state["original_context"],
        #         state["question"]
        #     )
        #     semantic_drift = measure_semantic_drift(
        #         original_understanding,
        #         current_understanding,
        #         state["ground_truth_answer"]
        #     )
        # else:
        #     # Subsequent hops: calculate based on answer correctness
        #     # If answer changed from correct to wrong = drift occurred
        #     semantic_drift = 0.0  # Will be calculated in app.py based on judge verdict

        original_understanding = extract_agent_understanding(
            state["original_context"],
            state["question"]
        )
        
        semantic_drift = measure_semantic_drift(
            original_understanding,
            current_understanding,
            state["ground_truth_answer"]
        )

        return {
            "current_text": rewritten_text,
            "history": [rewritten_text],
            "current_depth": state["current_depth"] + 1,
            # "token_preservation_scores": [preservation_score],
            "semantic_understanding_scores": [semantic_drift],
            "agent_answers": [agent_answer],
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
    return ChatOllama(model="gemma2:2b", temperature=0.1)


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


def LLM_Evaluator_Judge_3_QA_Answerable(rewritten_context: str, question: str, expected_answer: str,is_impossible: bool = False) -> int:
    """
    Judge 3: Can the question still be answered from the rewritten context?
    This directly validates task-relevant information preservation.
    
    Returns:
        1 if the question can be answered from rewritten context
        0 if the question cannot be answered (information lost)
    """
    if is_impossible:
        judge_llm = load_ollama_judge()
        
        system_prompt = SystemMessage(
            content=(
                "You are a reading comprehension AI. Answer the question based ONLY on the provided context. "
                "If the question cannot be answered from the context, respond with: 'UNANSWERABLE'"
            )
        )
        
        human_prompt = HumanMessage(
            content=f"""Context: {rewritten_context}

Question: {question}

Answer this question using ONLY information from the context above.
If the question cannot be answered from the context, respond with: UNANSWERABLE"""
        )
        
        try:
            response = judge_llm.invoke([system_prompt, human_prompt])
            judge_answer = response.content.strip().upper()
            return 1 if "UNANSWERABLE" in judge_answer else 0
        except Exception as e:
            print(f"Error in Judge 3 for impossible question: {e}")
            return 0
    else:
        judge_llm = load_ollama_judge()
        
        system_prompt = SystemMessage(
            content=(
                "You are a reading comprehension AI. Answer the question based ONLY on the provided context. "
                "If you cannot answer from the context, respond with: 'CANNOT_ANSWER'. "
                "Be concise and extract the key answer phrase only."
            )
        )
        
        human_prompt = HumanMessage(
            content=f"""Context: {rewritten_context}

            Question: {question}

            Answer this question using ONLY information from the context above.
            If the answer cannot be found in the context, respond with: CANNOT_ANSWER
            Be concise - provide only the answer phrase, no explanation."""
        )
        
        try:
            response = judge_llm.invoke([system_prompt, human_prompt])
            judge_answer = response.content.strip().lower()
            expected = expected_answer.lower().strip()
            
            if "cannot_answer" in judge_answer:
                return 0  
            elif expected in judge_answer or any(word in judge_answer for word in expected.split()):
                return 1  
            else:
                return 0  
        except Exception as e:
            print(f"Error in Judge 3: {e}")
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
