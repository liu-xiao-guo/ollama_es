import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()

# define init index
INIT_INDEX = os.getenv('INIT_INDEX', 'false').lower() == 'true'

# target url to scrape
TARGET_URL =  os.getenv('TARGET_URL', "https://open5gs.org/open5gs/docs/")

# http api port
HTTP_PORT = os.getenv('HTTP_PORT', 7654)

ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_ENDPOINT = os.getenv("ES_ENDPOINT")

elastic_index_name = "ollama_index"

url = f"https://{ES_USER}:{ES_PASSWORD}@{ES_ENDPOINT}:9200"

connection = Elasticsearch(
    hosts=[url], 
    ca_certs = "./http_ca.crt", 
    verify_certs = True
    )