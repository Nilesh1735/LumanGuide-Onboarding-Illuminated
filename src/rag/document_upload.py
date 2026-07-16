"""
Document ingestion module featuring a custom Semantic Chunker.
Splits text based on semantic distance spikes in sentence embeddings.
"""

import os
import re
import tempfile
import numpy as np
from typing import List

from fastapi import UploadFile, File, HTTPException
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.rag.retriever_setup import add_documents_to_retriever, embeddings
from src.tools.common_tools import enhance_description_with_llm
from src.security.secret_scanner import scan_document_for_secrets


class SemanticSentenceChunker:
    """
    Advanced chunker that segments text by analyzing semantic shifts 
    (Cosine Distance) between consecutive sentence embeddings.
    """
    def __init__(self, target_percentile: float = 95.0):
        self.target_percentile = target_percentile

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits raw text into sentences using regex boundary detection."""
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_endings.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _calculate_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Returns the mathematical Cosine Distance (1 - Cosine Similarity) between vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_1 = np.linalg.norm(vec1)
        norm_2 = np.linalg.norm(vec2)
        if norm_1 == 0 or norm_2 == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_1 * norm_2))

    def split_documents(self, documents: List[Document], embeddings_client) -> List[Document]:
        """
        Processes a list of Documents, embeds each sentence, and cuts boundaries 
        wherever semantic distance spikes above the target percentile threshold.
        """
        all_chunks = []
        
        for doc in documents:
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue
                
            # Embed all sentences in the document
            embeddings = embeddings_client.embed_documents(sentences)
            
            # Calculate distance between consecutive sentences
            distances = []
            for i in range(len(embeddings) - 1):
                dist = self._calculate_cosine_distance(
                    np.array(embeddings[i]), 
                    np.array(embeddings[i+1])
                )
                distances.append(dist)
            
            # Calculate static threshold based on target percentile
            if distances:
                threshold = np.percentile(distances, self.target_percentile)
            else:
                threshold = 1.0
                
            current_chunk = []
            for idx, sentence in enumerate(sentences):
                current_chunk.append(sentence)
                # If we cross the distance threshold, slice a new chunk
                if idx < len(distances) and distances[idx] > threshold:
                    all_chunks.append(Document(
                        page_content=" ".join(current_chunk),
                        metadata=doc.metadata.copy()
                    ))
                    current_chunk = []
                    
            if current_chunk:
                all_chunks.append(Document(
                    page_content=" ".join(current_chunk),
                    metadata=doc.metadata.copy()
                ))
                
        return all_chunks


def documents(description: str, file: UploadFile = File(...), tenant_id: str = "default_tenant"):
    """
    Process and upload a document for RAG using a semantic chunking pipeline.

    Args:
        description: User-provided document description.
        file: The uploaded file (PDF or TXT).
        tenant_id: Target tenant partition for secure vector storage.

    Returns:
        Boolean indicating upload success.
    """
    filename = file.filename
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Only PDF and TXT files are processed."
        )

    file_bytes = file.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract document stream: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Enhance description and store configuration. If the LLM is unavailable,
    # keep the user's description so upload can still proceed.
    try:
        description_llm = enhance_description_with_llm(description)
    except Exception:
        description_llm = description
    with open("description.txt", "w", encoding="utf-8") as f:
        f.write(description_llm)

    # Security: scan extracted text for hardcoded secrets (AWS keys, Slack
    # tokens, JWTs, private keys) and redact them in-place before chunking
    # and embedding. This prevents credentials from entering the FAISS
    # vector store where they could be surfaced to end users via retrieval.
    for doc in docs:
        scan_result = scan_document_for_secrets(doc.page_content)
        if scan_result.secrets_found:
            doc.page_content = scan_result.text

    # Process document using Semantic Sentence Chunker
    chunker = SemanticSentenceChunker(target_percentile=92.0)
    chunks = chunker.split_documents(docs, embeddings)

    # Inject multi-tenant tracking metadata into each chunk
    for chunk in chunks:
        chunk.metadata["tenant_id"] = tenant_id

    # Index chunks into the shared FAISS vector store.
    return add_documents_to_retriever(chunks, tenant_id=tenant_id)
