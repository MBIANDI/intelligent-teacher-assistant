import logging
import os
from typing import Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore

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
    chunks: list, embeddings, db_path: str
) -> Chroma:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        # collection_name="teacher_assistant_data",
    )
    return db



def parent_document_retriever_func(
    data_path: str,
    model_name: str,
    parent_chunk_size: int = 2000, # Chunk plus grand pour le contexte
    child_chunk_size: int = 400,   # Chunk petit pour la précision
    chunk_overlap: int = 50,
    db_path: str = "chroma_db",
    store_path: str = "parent_store",
    USE_OPENAI_EMBEDDINGS: bool = False,
):
    if not os.path.exists(data_path):
        logger.error(f"The data path {data_path} does not exist.")
        return None

    # 1. Chargement des documents originaux (non découpés)
    documents = data_loading(data_path)
    logger.info(f"Loaded {len(documents)} full documents.")

    # 2. Initialisation des Embeddings
    if USE_OPENAI_EMBEDDINGS:
        embeddings = openIAI_embedding_initialization(model_name)
    else:
        embeddings = embedding_initialization(model_name)

    # 3. Configuration des splitters
    # On définit comment on découpe les parents et les enfants
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # 4. Initialisation du VectorStore (Stocke les vecteurs des ENFANTS)
    vectorstore = Chroma(
        collection_name="split_parents",
        embedding_function=embeddings,
        persist_directory=db_path
    )

    # 5. Initialisation du DocStore persistant (Stocke le texte des PARENTS)
    # LocalFileStore permet de garder les parents sur le disque
    fs = LocalFileStore(os.path.join(store_path))
    store = create_kv_docstore(fs)

    # 6. Création du Parent Document Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 7. Ajout des documents au retriever
    retriever.add_documents(documents, ids=None)
    
    logger.info("Parent Document Retriever created and indexed.")
    return retriever


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

def init_retriever(
    data_path: str,
    model_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 400,
    db_path: str = "chroma_db",
    USE_OPENAI_EMBEDDINGS: bool = False,
    retriever_type: Literal["parent", "standard"] = "standard",
):
    if retriever_type == "parent":
        retriever = parent_document_retriever_func(
            data_path=data_path,
            model_name=model_name,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            db_path=db_path,
            store_path="parent_store",
            USE_OPENAI_EMBEDDINGS=USE_OPENAI_EMBEDDINGS,
        )
    
    else:

        db = vectorial_db_func(
            data_path=data_path,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db_path=db_path,
            USE_OPENAI_EMBEDDINGS=USE_OPENAI_EMBEDDINGS,
        )
        retriever = db.as_retriever()
    return retriever