from dotenv import load_dotenv

import openai
import os
load_dotenv()

import logging
import sys
import os

import numexpr as ne

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.node_parser import SentenceWindowNodeParser


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.api_key = os.getenv('OPENAI_API_KEY')


os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'


from llama_index.vector_stores import ElasticsearchStore

# Indexing with ES 
def indexing_data():
    #Load ES
    vector_store = ElasticsearchStore(
        index_name="law_bot",
        es_cloud_id="a360a60c18784a4288ef610006c3b861:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDAwZGM4M2JiYjU3NjRjZTliZDJlYjEyNTAwNTA2N2MxJDQzOTI5MzIyNGNlMjRiZDZhOTRkODYzOWQyZTNlYWJl",
        es_api_key="bUR1b3lZMEIzSUxOY1MxYjRvMEQ6ZE9PMS01UGlSSVdvdEhncUVkWmlWQQ=="
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    print("Load ES")
    return index

    # # Indexing and store ES
    # documents = SimpleDirectoryReader('../data/law/').load_data()
    # vector_store = ElasticsearchStore(
    #     index_name="law_bot",
    #     es_cloud_id="a360a60c18784a4288ef610006c3b861:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDAwZGM4M2JiYjU3NjRjZTliZDJlYjEyNTAwNTA2N2MxJDQzOTI5MzIyNGNlMjRiZDZhOTRkODYzOWQyZTNlYWJl",
    #     es_api_key="bUR1b3lZMEIzSUxOY1MxYjRvMEQ6ZE9PMS01UGlSSVdvdEhncUVkWmlWQQ=="
    # )
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
    # # create the sentence window node parser w/ default settings
    # node_parser = SentenceWindowNodeParser.from_defaults(
    #     window_size=3,
    #     window_metadata_key="window",
    #     original_text_metadata_key="original_text",
    # )
    # sentence_nodes = node_parser.get_nodes_from_documents(documents)
        
    # index = VectorStoreIndex(sentence_nodes, storage_context=storage_context)
    # print("Save ES")
    # return index
