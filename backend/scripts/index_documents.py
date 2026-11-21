"""
Local Document Indexer for Brand Guardian Compliance QA Pipeline.

Reads regulatory PDFs from backend/data, chunks them, and stores embeddings
in a persistent local ChromaDB vector store using local ONNX embeddings (no paid APIs).
"""

import os
import glob
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("local-indexer")

def index_docs():
    """
    Reads PDFs from backend/data, chunks them, and saves embeddings to local ChromaDB.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    data_folder = os.path.join(project_root, "backend", "data")
    
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "backend/data/chroma_db")
    if not os.path.isabs(persist_dir):
        persist_dir = os.path.join(project_root, persist_dir)
        
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "compliance_rules")

    logger.info("=" * 60)
    logger.info("Local Compliance Document Indexing (ChromaDB + Local ONNX MiniLM)")
    logger.info(f"Data directory:    {data_folder}")
    logger.info(f"Chroma persist dir: {persist_dir}")
    logger.info(f"Collection name:    {collection_name}")
    logger.info("=" * 60)

    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_folder}. Please add regulatory PDFs.")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process: {[os.path.basename(f) for f in pdf_files]}")

    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for pdf_path in pdf_files:
        try:
            filename = os.path.basename(pdf_path)
            logger.info(f"Loading and splitting: {filename}...")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()
            splits = text_splitter.split_documents(raw_docs)

            for split in splits:
                split.metadata["source"] = filename

            all_splits.extend(splits)
            logger.info(f" -> Created {len(splits)} chunks from {filename}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")

    if not all_splits:
        logger.warning("No document chunks were created.")
        return

    logger.info(f"Total chunks created: {len(all_splits)}. Storing into ChromaDB...")

    # Local Chroma ONNX MiniLM Embedding Function (Runs 100% locally on CPU)
    chroma_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Create Chroma vector store wrapper compatible with LangChain
    class LocalChromaEmbeddings:
        def __init__(self, ef):
            self.ef = ef
        def embed_documents(self, texts):
            return self.ef(texts)
        def embed_query(self, text):
            return self.ef([text])[0]

    embedding_wrapper = LocalChromaEmbeddings(chroma_embedding_fn)

    os.makedirs(persist_dir, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_wrapper
    )

    # Add documents to Chroma
    vector_store.add_documents(documents=all_splits)

    logger.info("=" * 60)
    logger.info(f"✅ Local Indexing Complete! {len(all_splits)} chunks stored in ChromaDB.")
    logger.info(f"Persistent location: {persist_dir}")

    # Verification query
    logger.info("Running verification query for 'disclosure guidelines'...")
    results = vector_store.similarity_search("disclosure guidelines", k=2)
    for i, r in enumerate(results, 1):
        logger.info(f"Result #{i} from [{r.metadata.get('source')}]: {r.page_content[:120]}...")
    logger.info("=" * 60)

if __name__ == "__main__":
    index_docs()
