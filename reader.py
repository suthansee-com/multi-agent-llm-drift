from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from configuration import READER_MODEL


def extract_agent_understanding(context: str, question: str, seed: int = 42) -> dict:
    llm = ChatOllama(model=READER_MODEL, temperature=0.0, seed=seed)

    system_prompt = SystemMessage(
        content=(
            "You are a precise extractive question-answering system.\n"
            "You answer only with short spans copied verbatim from the passage."
        )
    )

    human_prompt = HumanMessage(content=f"""Text: {context}

Question: {question}

Instructions:
- Answer with the SHORTEST span copied EXACTLY from the text above.
- Copy the wording verbatim. Do not paraphrase or write a full sentence.
- Do not restate the question or add explanation.
- If the text does not contain the answer, respond with exactly: CANNOT_ANSWER

Format your response exactly as:
KEY_FACTS: [facts from the text relevant to the question]
ANSWER: [span only]""")

    try:
        response = llm.invoke([system_prompt, human_prompt])
        response_text = response.content.strip()

        understanding = {"full_response": response_text, "key_facts": "", "answer": ""}

        if "KEY_FACTS:" in response_text:
            facts_start = response_text.find("KEY_FACTS:") + len("KEY_FACTS:")

            if "ANSWER:" in response_text:
                facts_end = response_text.find("ANSWER:")
            else:
                facts_end = len(response_text)

            understanding["key_facts"] = response_text[facts_start:facts_end].strip()

        if "ANSWER:" in response_text:
            answer_start = response_text.find("ANSWER:") + len("ANSWER:")
            understanding["answer"] = response_text[answer_start:].strip()

        return understanding
    except Exception as e:
        print(f"Error extracting understanding: {e}")
        return {"full_response": "", "key_facts": "", "answer": ""}
