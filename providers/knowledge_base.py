from functools import lru_cache
import os
from typing import List
import uuid

import arxiv
from docling.document_converter import DocumentConverter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from providers.providers import get_vectorstore
from utils.chainlit_msg import send_chainlit_message


class KnowledgeBase:
    def __init__(self, topic: str):
        self.collection_name = f"{uuid.uuid4().hex}"
        self.vector_store = get_vectorstore(self.collection_name)
        self.converter = DocumentConverter()

        # Configure Markdown Header Splitter
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Configure Recursive Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.processed_ids = set()

        # Load existing processed IDs to for deduplication
        try:
            existing_data = self.vector_store.get()
            if existing_data and "metadatas" in existing_data:
                for meta in existing_data["metadatas"]:
                    if "source" in meta:
                        self.processed_ids.add(meta["source"].split('/')[-1])
        except Exception as e:
            logger.warning(f"Could not load existing IDs: {e}")

    def download_and_process(self, papers: List[arxiv.Result]):
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        new_docs_count = 0
        for i, paper in enumerate(papers):
            paper_id = paper.entry_id.split('/')[-1]
            send_chainlit_message(f"🧐 **Downloading and Reading Paper ({i+1}/{len(papers)}):** *{paper.title}*")
            if paper_id in self.processed_ids:
                logger.info(f"Skipping existing paper: {paper.title}")
                continue

            pdf_path = os.path.join("downloads", f"{paper_id}.pdf")
            try:
                if not os.path.exists(pdf_path):
                    logger.info(f"Downloading PDF: {paper.title}")
                    paper.download_pdf(dirpath="downloads", filename=f"{paper_id}.pdf")

                logger.info(f"Docling processing: {pdf_path}")

                conversion_result = self.converter.convert(pdf_path)
                markdown_text = conversion_result.document.export_to_markdown()

                header_splits = self.markdown_splitter.split_text(markdown_text)

                base_metadata = {
                    "source": paper.entry_id,
                    "title": paper.title,
                    "year": paper.published.year
                }
                for doc in header_splits:
                    doc.metadata.update(base_metadata)

                filtered_splits = []
                for doc in header_splits:
                    h1 = doc.metadata.get("Header 1", "").lower()
                    h2 = doc.metadata.get("Header 2", "").lower()
                    if "reference" in h1 or "references" in h1 or "bibliography" in h1 or "reference" in h2 or "references" in h2 or"bibliography" in h2:
                        continue
                    doc.metadata.update(base_metadata)
                    filtered_splits.append(doc)

                chunks = filtered_splits

                if chunks:
                    self.vector_store.add_documents(chunks)
                    self.processed_ids.add(paper_id)
                    new_docs_count += len(chunks)
                    logger.info(f"Added {len(chunks)} chunks from paper: {paper.title}")

            except Exception as e:
                logger.error(f"Error processing {paper.title}: {e}")
        return new_docs_count

    def retrieve(self, query: str, k=5):
        return self.vector_store.similarity_search(query, k=k)

@lru_cache(maxsize=10)
def get_knowledge_base(topic: str) -> KnowledgeBase:
    return KnowledgeBase(topic)
