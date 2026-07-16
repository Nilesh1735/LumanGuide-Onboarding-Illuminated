import os
import tempfile
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

def documents(description: str, file: UploadFile) -> bool:
    """
    Synchronous document upload function.
    Matches the call signature in routes.py: documents(description, file)
    """
    try:
        # 1. Read file contents synchronously
        contents = file.file.read()
        
        # 2. Extract text based on file type
        text = ""
        if file.filename and file.filename.lower().endswith('.pdf'):
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            text = contents.decode('utf-8', errors='ignore')
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        # 3. Scan for secrets
        from src.security.secret_scanner import scan_document_for_secrets
        scan_result = scan_document_for_secrets(text)
        text = scan_result.text

        # 4. Chunk and add to FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from src.rag.retriever_setup import add_documents_to_retriever
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata={"source": file.filename, "description": description}) for chunk in chunks]
        
        add_documents_to_retriever(docs)
        
        # 5. Write description to a SAFE temporary directory (Fixes Render 500 error)
        temp_dir = tempfile.gettempdir()
        desc_path = os.path.join(temp_dir, "description.txt")
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(description)
            
        logger.info(f"Successfully indexed {len(docs)} chunks from {file.filename}")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to upload document")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")