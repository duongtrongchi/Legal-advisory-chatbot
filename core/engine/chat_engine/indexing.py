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


def indexing_data():
    try:
        storage_context = StorageContext.from_defaults(persist_dir='../assets/cache/law/')
        sentence_index = load_index_from_storage(storage_context)
        print('loading from disk')
        return sentence_index
    except:
        documents = SimpleDirectoryReader('../data/law/').load_data()

        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        sentence_nodes = node_parser.get_nodes_from_documents(documents)
        sentence_index = VectorStoreIndex(sentence_nodes)
        sentence_index.storage_context.persist('../assets/cache/law/')
        print('persisting to disk')
        return sentence_index