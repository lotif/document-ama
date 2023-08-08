import datetime
import hashlib
import logging
import os
import requests
import shutil
import sys
import traceback

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "RETURN_SOURCE_DOCUMENTS": True,
    "VECTOR_COUNT": 2,
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "DATA_PATH": "data",
    "MODELS_PATH": "models",
    # MODEL_TYPE: 'mpt'
    # MODEL_BIN_PATH: 'models/mpt-7b-instruct.ggmlv3.q8_0.bin'
    "MODEL_TYPE": "llama",
    "MODEL_BIN_PATH": "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    "EMBEDDINGS_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "MAX_NEW_TOKENS": 256,
    "TEMPERATURE": 0.01,
}

MODEL_URLS = {
    "models/llama-2-7b-chat.ggmlv3.q8_0.bin": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin",
}

QA_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


@st.cache_resource(show_spinner=False)
def init_resources():
    resources = {}

    with st.spinner("Initializing models..."):
        logger.info("Initializing models...")
        # Local CTransformers model
        resources["llm"] = CTransformers(
            model=CONFIG["MODEL_BIN_PATH"],
            model_type=CONFIG["MODEL_TYPE"],
            config={
                'max_new_tokens': CONFIG["MAX_NEW_TOKENS"],
                'temperature': CONFIG["TEMPERATURE"],
            },
        )
        resources["text_splitter"] = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
        )
        resources["embeddings_model"] = HuggingFaceEmbeddings(
            model_name=CONFIG["EMBEDDINGS_MODEL"],
            model_kwargs={"device": "cpu"},
        )
        logger.info("Done initializing models.")

    return resources


def main():
    init_folders()
    error = download_model_if_necessary()
    if error is not None:
        st.error(error)
        logger.error(error)
        return None

    resources = init_resources()

    st.header("Document AMA")

    uploaded_file = st.file_uploader("Upload a .pdf or .txt document")

    if uploaded_file is None:
        return

    with st.spinner("Processing file..."):
        faiss_path, error = process_file(uploaded_file, resources["text_splitter"], resources["embeddings_model"])

        if faiss_path is None and error is not None:
            if type(error) == Exception:
                st.exception(error)
                logger.exception(error)
            else:
                st.error(error)
                logger.error(error)
            return

    with st.form(key="question_form"):
        question = st.text_input("Ask your question:")
        submit_button = st.form_submit_button(label="Ask")

        if not submit_button:
            return

        if question is None or len(question) == 0:
            st.error("Please enter a valid question.")
            return

        with st.spinner("Retrieving answer..."):
            logger.info("Started retrieving answer...")
            start = datetime.datetime.now()

            dbqa = setup_dbqa(faiss_path, resources["llm"], resources["embeddings_model"])
            response = dbqa({"query": question})

            end = datetime.datetime.now()
            delta = (end - start)
            logger.info(f"Answer retrieved in {delta.seconds}s")

        if response is None or type(response) is not dict or "result" not in response:
            st.error(f"Response is invalid: {response}")
            return

        st.markdown(f"""
            <p style='background-color: rgb(221, 255, 192);padding: 20px;'>
                {response['result']}
            </p>
        """, unsafe_allow_html=True)

        for document in response["source_documents"]:
            page = "N/A"
            if "page" in document.metadata:
                page = str(document.metadata["page"] + 1)
            st.write(f"**Page {page}:**")
            st.write(document.page_content)
            st.divider()

        st.write(f"Response retrieved in {delta.seconds}s")
        st.markdown(
            f"Q&A Model: {CONFIG['MODEL_BIN_PATH']}" +
            "<br/>" +
            f"Embeddings Model: {CONFIG['EMBEDDINGS_MODEL']}",
            unsafe_allow_html=True,
        )


def init_folders():
    if not os.path.exists(CONFIG["DATA_PATH"]):
        os.mkdir(CONFIG["DATA_PATH"])
    if not os.path.exists(CONFIG["MODELS_PATH"]):
        os.mkdir(CONFIG["MODELS_PATH"])


def download_model_if_necessary():
    model_path = CONFIG["MODEL_BIN_PATH"]
    if not os.path.exists(model_path):
        if not model_path in MODEL_URLS:
            return f"Model {model_path} is not loaded and does not have a URL assigned."

        logger.info("Downloading model...")

        st.write("Downloading the model. This might take a while, do not close or refresh the window.")
        progress_bar = st.progress(0.0, text="Starting download...")

        with open(model_path, "wb") as model_file:
            response = requests.get(MODEL_URLS[model_path], allow_redirects=True, stream=True)
            total_length = response.headers.get("content-length")
            if total_length is None:  # no content length header
                model_file.write(response.content)
            else:
                def format_progress_data(progress_data):
                    return f"{((progress_data / 1024) / 1024) / 1024:0.2f}GB"

                downloaded = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    model_file.write(data)
                    percent = (100 * downloaded) / total_length
                    progress_bar.progress(
                        percent / 100,
                        text=f"{percent:.1f}% ({format_progress_data(downloaded)}/{format_progress_data(total_length)})"
                    )

        logger.info("Done downloading model.")


def process_file(uploaded_file, text_splitter, embeddings_model):
    try:
        logger.info("Started processing file...")

        extension = os.path.splitext(uploaded_file.name)[-1].lower()
        if extension != ".pdf" and extension != ".txt":
            return None, f"File type {extension} not supported. Please upload a '.pdf' or '.txt' file."

        file_bytes = uploaded_file.read()
        hash = hashlib.sha256(file_bytes).hexdigest()
        data_path = f"{CONFIG['DATA_PATH']}/{hash}"
        file_path = f"{data_path}/source_file{extension}"
        faiss_path = f"{data_path}/faiss"

        if os.path.exists(faiss_path):
            logger.info("Found file in cache.")
            return faiss_path, None

        os.mkdir(data_path)
        os.mkdir(faiss_path)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        if extension == ".pdf":
            loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        else:
            loader = DirectoryLoader(data_path, glob='*.txt', loader_cls=TextLoader)

        documents = loader.load()
        texts = text_splitter.split_documents(documents)

        if len(texts) == 0:
            raise Exception("Could not extract text from file.")

        vectorstore = FAISS.from_documents(texts, embeddings_model)
        vectorstore.save_local(faiss_path)

        logger.info("Done processing file.")

        return faiss_path, None

    except Exception as e:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        print(traceback.format_exc())
        return None, e


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": CONFIG["VECTOR_COUNT"]}),
        return_source_documents=CONFIG["RETURN_SOURCE_DOCUMENTS"],
        chain_type_kwargs={"prompt": prompt}
    )
    return dbqa


def setup_dbqa(faiss_path, llm, embeddings_model):
    vectordb = FAISS.load_local(faiss_path, embeddings_model)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


main()
