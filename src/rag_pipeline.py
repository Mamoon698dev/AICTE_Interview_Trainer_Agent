"""
rag_pipeline.py – RAG pipeline built on LangChain + ChromaDB.

Responsibilities
----------------
* Ingest documents (PDFs, DOCX, plain text) into a persistent vector store.
* Retrieve the most relevant chunks for a given query.
* Expose a RAG chain that combines retrieval with IBM Granite generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

from src.config import config
from src.watsonx_client import get_llm, get_embeddings

logger = logging.getLogger(__name__)

# ── Collection names ──────────────────────────────────────────────────────────
COLLECTION_KB = "interview_knowledge_base"
COLLECTION_RESUME = "user_resumes"


class RAGPipeline:
    """End-to-end RAG pipeline for the Interview Trainer Agent."""

    def __init__(self) -> None:
        config.ensure_dirs()
        self._embeddings = get_embeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._kb_store: Chroma | None = None
        self._resume_store: Chroma | None = None
        self._load_existing_stores()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_existing_stores(self) -> None:
        persist = config.CHROMA_PERSIST_DIR
        try:
            self._kb_store = Chroma(
                collection_name=COLLECTION_KB,
                embedding_function=self._embeddings,
                persist_directory=persist,
            )
            self._resume_store = Chroma(
                collection_name=COLLECTION_RESUME,
                embedding_function=self._embeddings,
                persist_directory=persist,
            )
            logger.info("Loaded existing ChromaDB collections from %s", persist)
        except Exception as exc:
            logger.warning("Could not load existing collections: %s", exc)

    def _split_and_store(
        self, docs: list, store: Chroma, collection_name: str
    ) -> Chroma:
        chunks = self._splitter.split_documents(docs)
        logger.info(
            "Splitting %d docs → %d chunks → '%s'",
            len(docs),
            len(chunks),
            collection_name,
        )
        if store is None:
            store = Chroma.from_documents(
                chunks,
                embedding=self._embeddings,
                collection_name=collection_name,
                persist_directory=config.CHROMA_PERSIST_DIR,
            )
        else:
            store.add_documents(chunks)
        return store

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_knowledge_base(self, directory: str | Path | None = None) -> int:
        """Load all PDFs / DOCX / TXT files from the knowledge-base directory."""
        kb_dir = Path(directory or config.KB_DIR)
        if not kb_dir.exists():
            logger.warning("Knowledge-base directory not found: %s", kb_dir)
            return 0

        loaders = {
            "**/*.pdf": PyPDFLoader,
            "**/*.docx": Docx2txtLoader,
            "**/*.txt": TextLoader,
            "**/*.md": TextLoader,
        }
        docs: list = []
        for glob_pattern, loader_cls in loaders.items():
            for fp in kb_dir.glob(glob_pattern):
                try:
                    docs.extend(loader_cls(str(fp)).load())
                    logger.debug("Loaded: %s", fp)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", fp, exc)

        if not docs:
            logger.info("No documents found in %s", kb_dir)
            return 0

        self._kb_store = self._split_and_store(docs, self._kb_store, COLLECTION_KB)
        return len(docs)

    def ingest_resume(self, file_path: str | Path) -> bool:
        """Parse and index a single resume (PDF, DOCX, or TXT)."""
        fp = Path(file_path)
        if not fp.exists():
            logger.error("Resume file not found: %s", fp)
            return False

        suffix = fp.suffix.lower()
        loader_map = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
        }
        loader_cls = loader_map.get(suffix)
        if not loader_cls:
            logger.error("Unsupported file type: %s", suffix)
            return False

        try:
            docs = loader_cls(str(fp)).load()
            # Tag each chunk with source metadata
            for d in docs:
                d.metadata["source"] = "resume"
                d.metadata["file"] = fp.name
            self._resume_store = self._split_and_store(
                docs, self._resume_store, COLLECTION_RESUME
            )
            logger.info("Resume ingested: %s (%d pages/sections)", fp.name, len(docs))
            return True
        except Exception as exc:
            logger.error("Failed to ingest resume %s: %s", fp, exc)
            return False

    def retrieve(
        self, query: str, collection: str = "kb", k: int | None = None
    ) -> list:
        """Return the top-k most relevant document chunks for a query."""
        k = k or config.RAG_TOP_K
        store = self._kb_store if collection == "kb" else self._resume_store
        if store is None:
            return []
        try:
            return store.similarity_search(query, k=k)
        except Exception as exc:
            logger.warning("Retrieval error: %s", exc)
            return []

    def build_qa_chain(self, collection: str = "kb"):
        """Build a LangChain LCEL RAG chain over the chosen collection.

        Returns a runnable that accepts {"query": str} and returns a str answer.
        """
        store = self._kb_store if collection == "kb" else self._resume_store
        if store is None:
            raise RuntimeError(
                f"Collection '{collection}' is empty. Ingest documents first."
            )

        prompt = PromptTemplate(
            template=(
                "You are an expert interview coach. Use the context below to "
                "answer the question. If the context does not contain enough "
                "information, say so and give general best-practice advice.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        retriever = store.as_retriever(search_kwargs={"k": config.RAG_TOP_K})

        def _format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        # LCEL chain: retrieve → format → prompt → LLM → parse
        chain = (
            RunnableMap({
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            })
            | prompt
            | get_llm()
            | StrOutputParser()
        )
        return chain

    def get_context_string(self, query: str, collection: str = "kb") -> str:
        """Return retrieved chunks merged into a single string."""
        docs = self.retrieve(query, collection=collection)
        return "\n\n---\n\n".join(d.page_content for d in docs)
