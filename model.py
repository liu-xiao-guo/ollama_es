from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from elasticsearch import Elasticsearch, helpers

from config import *
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global conversation
conversation = None

global vectordb
vectordb = None

def init_index():
    global vectordb
    
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return
    
    # scrape data from web
    documents = RecursiveUrlLoader(
        TARGET_URL,
        max_depth=4,
        extractor=lambda x: Soup(x, "html.parser").text,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True,
        # drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
    ).load()

    logging.info("index creating with `%d` documents", len(documents))

    # split text
    # this chunk_size and chunk_overlap effects to the prompt size
    # execeed promt size causes error `prompt size exceeds the context window size and cannot be processed`
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    
    # create embeddings with huggingface embedding model `all-MiniLM-L6-v2`
    # then persist the vector index on vector db
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not connection.indices.exists(index=elastic_index_name):
        print("The index does not exist, going to generate embeddings")   
        vectordb = ElasticsearchStore.from_documents( 
                documents,
                embedding = embeddings, 
                es_url = url, 
                es_connection = connection,
                index_name = elastic_index_name, 
                es_user = ES_USER,
                es_password = ES_PASSWORD
        )
    else: 
        print("The index already existed")    
        vectordb = ElasticsearchStore(
            es_connection = connection,
            embedding = embeddings,
            es_url = url, 
            index_name = elastic_index_name, 
            es_user = ES_USER,
            es_password = ES_PASSWORD    
        )   

def init_conversation():
    global conversation
    global vectordb

    # llama2 llm which runs with ollama
    # ollama expose an api for the llam in `localhost:11434`
    llm = Ollama(
        model="llama2",
        base_url="http://localhost:11434",
        verbose=True,
    )

    # create conversation
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever = vectordb.as_retriever(),
        return_source_documents = True,
        verbose = True,
    )

def chat(question, user_id):
    global conversation

    chat_history = []
    response = conversation({"question": question, "chat_history": chat_history})
    answer = response['answer']

    logging.info("got response from llm - %s", answer)

    # TODO save history

    return answer