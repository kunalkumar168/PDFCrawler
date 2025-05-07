import os
import warnings
from pathlib import Path
from uuid import uuid4

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_and_split_pdfs(pdf_path: Path):
    """Load a PDF file and split it into text chunks."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ".", "|"],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    return chunks

def process_multiple_pdfs(pdf_paths):
    """Extract text and metadata from a list of PDF paths."""
    documents = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        chunks = load_and_split_pdfs(pdf_path)
        for chunk in chunks:
            page_content = chunk.page_content.lower()
            metadata = {
                "source": chunk.metadata.get("source", str(pdf_path.name)),
                "page": chunk.metadata.get("page", -1),
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
    print(f"Total chunks processed: {len(documents)}")
    return documents

def create_vectorstore(documents, embedding_model_name: str):
    """Create a FAISS vector store from documents using a HuggingFace embedding model."""
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    return vector_store

def main():
    current_workdir = Path.cwd()
    files_folder = f"{current_workdir}/files"
    vectordb_path = f"{current_workdir}/vectordb" 
    file_paths = list(Path(files_folder).rglob("*.pdf"))
    if not file_paths:
        print(f"[ERROR] No PDF files found in {files_folder}")
        return

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    documents = process_multiple_pdfs(file_paths)
    vectorstore = create_vectorstore(documents, embedding_model_name)
    vectorstore.save_local(str(vectordb_path))
    print(f"Vector DB saved to: {vectordb_path}")


if __name__ == "__main__":
    main()