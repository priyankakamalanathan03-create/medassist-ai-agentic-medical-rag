from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
