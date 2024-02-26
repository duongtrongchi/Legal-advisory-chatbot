from dotenv import load_dotenv

import openai
import os
os.environ["OPENAI_API_KEY"] = "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"
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

def is_folder_empty(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return True

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Check if the folder is empty
    if not files:
        print(f"The folder '{folder_path}' is empty.")
        return True
      
    print(f"The folder '{folder_path}' contains files: {files}")
    return False

import shutil

def move_file(source_path, destination_path):
    try:
        # Move the file from the source to the destination
        shutil.move(source_path, destination_path)
        print(f"File moved successfully from '{source_path}' to '{destination_path}'.")
    except FileNotFoundError:
        print(f"Error: The file at '{source_path}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")

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
    print("="*100)
   
    # Check new update document
    folder_path = '../data/law/upload/'
    if not is_folder_empty(folder_path):
      
        print("Add node to ES")
        documents = SimpleDirectoryReader(folder_path).load_data()
      
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        new_nodes = node_parser.get_nodes_from_documents(documents)
        index.insert_nodes(new_nodes)
      
        files = os.listdir(folder_path)
        for file in files: 
            source_path = f'../data/law/upload/{file}'
            destination_path = '../data/law/'
            move_file(source_path, destination_path)
   
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
