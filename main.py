__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import os
import tempfile
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
load_dotenv()

st.title("ChatPDF")
st.write("---")

openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

button(username="gauguin135", floating=True, width=221)

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key,
    )

    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    db = Chroma.from_documents(texts, embeddings_model)

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.container.markdown(self.text)

    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(), llm=llm
            )

            prompt = hub.pull("rlm/rag-prompt")

            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | generate_llm
                | StrOutputParser()
            )

            result = rag_chain.invoke(question)

