import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_extras.buy_me_a_coffee import button
import faiss

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.load import dumps, loads

button(username="gauguin135", floating=True, width=221)

st.title("ChatPDF with Multiquery+hybridSearch+RagFusion")
st.write("---")
st.write("PDF 파일을 업로드하고 내용을 기반으로 질문하세요.")

openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

model_choice = st.selectbox(
    '사용할 GPT 모델을 선택하세요:',
    ['gpt-4o-mini', 'gpt4.1-mini','gpt-4.1', 'gpt-5-mini', 'gpt-5']
)

uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load()
    return pages

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
    embedding_dimension = len(OpenAIEmbeddings(openai_api_key=openai_key).embed_query("hello world"))

    index = faiss.IndexFlatL2(embedding_dimension)
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vectorstore.add_documents(documents=splits, ids=range(len(splits)))

    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k':1, 'fetch_k': 4}
    )
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.2, 0.8]
    )

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.container.markdown(self.text)

    template = """
                당신은 AI 언어 모델 조수입니다. 목표는 주어진 사용자 질문과 관련해 벡터 데이터베이스에서 관련 문서를 
                검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다.
                사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다. 각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results[:top_n]
    
    retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion
    template = """
        다음 맥락을 바탕으로 질문에 답변하세요:
        {context}

        질문: {question}
    """


    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner('답변 생성 중...'):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])

            final_rag_chain = (
                {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            result = final_rag_chain.invoke(question)