import hashlib
import os

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from chatbot.cache import BotCacheManager
from chatbot.config import AppConfig, DEFAULT_CONFIG
from chatbot.embeddings import require_embed_model


def make_node_parser(chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def build_index_from_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> VectorStoreIndex:
    parser = make_node_parser(chunk_size, chunk_overlap)
    return VectorStoreIndex.from_documents(
        documents,
        transformations=[parser],
        embed_model=require_embed_model(),
        show_progress=False,
    )


def save_index(
    index: VectorStoreIndex,
    file_hash: str,
    cache_dir: str,
    cache_manager: BotCacheManager,
) -> bool:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, f"index_{file_hash}")
        index.storage_context.persist(persist_dir=index_path)
        cache_manager.cache_response(
            f"llamaindex_{file_hash}",
            {"index_path": index_path, "file_hash": file_hash, "status": "saved"},
        )
        return True
    except OSError:
        return False


def load_index(
    file_hash: str,
    cache_manager: BotCacheManager,
) -> VectorStoreIndex | None:
    try:
        cached = cache_manager.get_cached_response(f"llamaindex_{file_hash}")
        if not cached or "index_path" not in cached:
            return None
        index_path = cached["index_path"]
        if not os.path.isdir(index_path):
            return None
        storage = StorageContext.from_defaults(persist_dir=index_path)
        return load_index_from_storage(storage, embed_model=require_embed_model())
    except Exception:
        return None


def get_or_create_index(
    documents: list[Document],
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    config: AppConfig | None = None,
    cache_manager: BotCacheManager | None = None,
) -> VectorStoreIndex:
    config = config or DEFAULT_CONFIG
    cache_manager = cache_manager or BotCacheManager(cache_dir=config.bot_cache_dir)
    key = index_cache_key(file_hash, chunk_size, chunk_overlap, config)

    existing = load_index(key, cache_manager)
    if existing is not None:
        return existing

    index = build_index_from_documents(documents, chunk_size, chunk_overlap)
    save_index(index, key, config.cache_dir, cache_manager)
    return index


def index_cache_key(file_hash: str, chunk_size: int, chunk_overlap: int, config: AppConfig) -> str:
    """Vectors from different embedding models or chunkings must never be mixed."""
    embed_id = hashlib.sha256(config.embed_identity.encode()).hexdigest()[:10]
    return f"{file_hash}_{embed_id}_{chunk_size}_{chunk_overlap}"
