from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from chatbot.answering import FakeLLM, generate_answer
from chatbot.config import RetrievalSettings
from chatbot.prompts import REFUSAL_PHRASE


def _index_with_node(text: str, embed_model) -> VectorStoreIndex:
    node = TextNode(text=text, id_="n1", metadata={"filename": "policy.txt"})
    return VectorStoreIndex(nodes=[node], embed_model=embed_model)


def test_refusal_path_with_fake_llm(embed_model):
    index = _index_with_node("Only unrelated content about widgets.", embed_model)
    settings = RetrievalSettings(
        chunk_size=600,
        chunk_overlap=100,
        top_k=3,
        hybrid=False,
        rerank=False,
        temperature=0.0,
    )
    llm = FakeLLM(REFUSAL_PHRASE)
    result = generate_answer(index, "What is the CEO salary?", llm, settings)
    assert result.refused
    assert REFUSAL_PHRASE in result.text
