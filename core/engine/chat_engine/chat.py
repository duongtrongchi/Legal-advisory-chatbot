import logging
import sys
import os

import openai
from dotenv import load_dotenv
load_dotenv()

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores import ElasticsearchStore


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')


class ChatEngine:

    def __init__(self, documents_path="../../data/", new_indexing=False):
        self.vector_store = ElasticsearchStore(
                                es_url="http://localhost:9200",
                                index_name="law_index",
                            )
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        self.documents = SimpleDirectoryReader(documents_path).load_data()
        self.sentence_nodes = self.node_parser.get_nodes_from_documents(self.documents)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        if new_indexing:
            self.index = VectorStoreIndex(
                self.sentence_nodes,
                storage_context=self.storage_context,
            )


    def chat(self, ques="Bạn là ai?"):
        query_engine = self.index.as_query_engine()
        res =query_engine.query(ques)
        return res

