from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM

from chatbot.citations import (
    SourceCitation,
    citations_from_nodes,
    format_context_block,
    is_refusal_answer,
)
from chatbot.config import AppConfig, DEFAULT_CONFIG, RetrievalSettings
from chatbot.prompts import FAITHFULNESS_JUDGE_TEMPLATE, GROUNDED_QA_TEMPLATE, REFUSAL_PHRASE
from chatbot.retrieval import CrossEncoderReranker, build_retriever


@dataclass
class ChatAnswer:
    text: str
    citations: list[SourceCitation]
    faithfulness: str | None  # "supported" | "unsupported" | None
    refused: bool


class FakeLLM:
    """Minimal LLM stub for unit tests."""

    def __init__(self, response: str):
        self.response = response

    def complete(self, prompt: str):
        class R:
            def __init__(self, text):
                self.text = text

        return R(self.response)


def generate_answer(
    index: VectorStoreIndex,
    question: str,
    llm: LLM,
    settings: RetrievalSettings,
    config: AppConfig | None = None,
    faithfulness_check: bool = False,
    reranker: CrossEncoderReranker | None = None,
) -> ChatAnswer:
    config = config or DEFAULT_CONFIG

    def reranker_factory():
        nonlocal reranker
        if reranker is None:
            reranker = CrossEncoderReranker(config.cross_encoder_model)
        return reranker

    retrieve = build_retriever(index, settings, reranker_factory=reranker_factory)
    nodes = retrieve(question)
    citations = citations_from_nodes(nodes)
    context = format_context_block(citations, nodes)

    prompt = GROUNDED_QA_TEMPLATE.format(
        context=context,
        question=question,
        refusal=REFUSAL_PHRASE,
    )
    response = llm.complete(prompt)
    answer_text = getattr(response, "text", str(response)).strip()

    faithfulness = None
    if faithfulness_check and not is_refusal_answer(answer_text, REFUSAL_PHRASE):
        judge_prompt = FAITHFULNESS_JUDGE_TEMPLATE.format(context=context, answer=answer_text)
        judge_raw = llm.complete(judge_prompt)
        judge_resp = getattr(judge_raw, "text", str(judge_raw)).strip().upper()
        if "UNSUPPORTED" in judge_resp:
            faithfulness = "unsupported"
        elif "SUPPORTED" in judge_resp:
            faithfulness = "supported"
        else:
            faithfulness = "unknown"

    return ChatAnswer(
        text=answer_text,
        citations=citations,
        faithfulness=faithfulness,
        refused=is_refusal_answer(answer_text, REFUSAL_PHRASE),
    )
