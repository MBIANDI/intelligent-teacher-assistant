import logging
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_loading(data_path: str) -> list:
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def text_chunking(
    documents: list, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def openIAI_embedding_initialization(model_name: str) -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(
        model=model_name,
    )
    return embeddings


def embedding_initialization(model_name: str) -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def create_vector_db(
    chunks: list, embeddings: HuggingFaceEmbeddings, db_path: str
) -> Chroma:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        # collection_name="teacher_assistant_data",
    )
    return db


def vectorial_db_func(
    data_path: str,
    model_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    db_path: str = "chroma_db",
    USE_OPENAI_EMBEDDINGS: bool = False,
):
    # Check if data_path exist
    if not os.path.exists(data_path):
        logger.error(f"The data path {data_path} does not exist.")
        return None
    # If db_path exists, remove it to create a new one
    # if os.path.exists(db_path):
    #     shutil.rmtree(db_path)
    #     logger.info(f"Existing database at {db_path} removed.")
    # Load data
    documents = data_loading(data_path)
    logger.info(f"Loaded {len(documents)} documents from {data_path}.")
    # Chunk text
    chunks = text_chunking(documents, chunk_size, chunk_overlap)
    logger.info(f"Created {len(chunks)} text chunks.")
    # Initialize embeddings
    if USE_OPENAI_EMBEDDINGS:
        embeddings = openIAI_embedding_initialization(model_name)
        logger.info(f"Initialized OpenAI embeddings using model {model_name}.")
    else:
        embeddings = embedding_initialization(model_name)
    logger.info(f"Initialized embeddings using model {model_name}.")
    # Create vector database
    db = create_vector_db(chunks, embeddings, db_path)
    logger.info(f"Vector database created at {db_path}.")
    return db
