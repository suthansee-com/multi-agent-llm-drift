AGENT_MODEL = "gemma2:2b"
READER_MODEL = "llama3:8b"
RATER_MODEL_A = "llama3:8b"
RATER_MODEL_B = "mistral:7b"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

faithfulness_rubric = (
    "Decide whether the REWRITTEN passage is FAITHFUL to the ORIGINAL source. "
    "It is FAITHFUL only if every factual claim it makes is supported by the "
    "original, with no invented facts, no dropped facts that change the meaning, "
    "and no distorted relationships between entities. If any claim is unsupported, "
    "invented, or contradicts the source, it is HALLUCINATED. "
    "Respond with ONLY one word: FAITHFUL or HALLUCINATED."
)