from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.memory import ConversationBufferMemory
import openai
import os
import logging
#---
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)

st.set_page_config(
    page_title="DocumentGPT",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=80,
    return_messages=True
)

def add_memory_message(input, output):
    memory.save_context({"input": input}, {"output": output})


def get_memory_history(_):
    return memory.load_memory_variables({})["history"]


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):

    file_path = f"./.cache/files/{file.name}"
    folder_path = os.path.dirname(file_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.info(f'folder_path: {folder_path}')
    logging.info(f'file_path: {file_path}')

    file_content = file.read()
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            history : {chat_history}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

def check_openai_api_key(key):
    try:
        openai.api_key = key
        openai.Model.list()
        return True
    except Exception:
        return False       


with st.sidebar:
    
    key = st.text_input("api key please!", type="password").strip()

    if check_openai_api_key(key):
        st.write("먹음")
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
    else:
        file =[]
        st.write("안먹음")

    st.link_button("Go To Git", "https://github.com/surluka/test", help=None, type="primary")

    code = open('app.py','r',encoding="UTF-8").read()
    st.code(code, language="python")



if file:
       
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "chat_history": RunnableLambda(get_memory_history),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

        add_memory_message(message, response.content)
        st.session_state["history_message"] += get_memory_history(message)
        # st.write("session state : ", st.session_state["history_message"])    


else:
    st.session_state["messages"] = []
    st.session_state["history_message"] = []