REFUSAL_PHRASE = "I could not find this in the documents"

GROUNDED_QA_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the numbered context sources below.

Rules:
1. Use only facts supported by the context. Cite sources inline using [1], [2], etc.
2. If the context does not contain enough information, reply exactly: "{refusal}"
3. Do not invent policies, prices, or procedures not present in the context.
4. Keep answers concise and factual.

Context:
{context}

Question: {question}

Answer (with citations):"""


FAITHFULNESS_JUDGE_TEMPLATE = """You are a strict judge. Given CONTEXT and ANSWER, decide if every claim in ANSWER is directly supported by CONTEXT.

Reply with a single word: SUPPORTED or UNSUPPORTED.

CONTEXT:
{context}

ANSWER:
{answer}
"""
