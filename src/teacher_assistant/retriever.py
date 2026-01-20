import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def init_llm(
    model_name: str, temperature: float = 1, api_key: str = os.getenv("OPENAI_API_KEY")
) -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        max_retries=3,
        request_timeout=120,
    )
    return llm


def retriever(llm: ChatOpenAI, prompt: str, vector_db, k: int = 4):
    admin_prompt = PromptTemplate(
        template=prompt, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": admin_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False,
    )
    return qa


def prof_assistant(
    llm: ChatOpenAI,
    prompt: str,
    retriever,
) -> ConversationalRetrievalChain:
    admin_prompt = PromptTemplate(
        template=prompt, input_variables=["context", "question"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
        
    prof_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": admin_prompt},
        get_chat_history=lambda h: h,
    )

    return prof_chain
