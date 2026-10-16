from llama_index.core import Document

from chatbot.indexing import make_node_parser


def test_sentence_splitter_produces_multiple_nodes():
    long_text = "word " * 400
    doc = Document(text=long_text, metadata={"filename": "t.txt"})
    parser = make_node_parser(chunk_size=200, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents([doc])
    assert len(nodes) > 1
