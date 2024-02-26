import logging
import sys
import os

import openai
from dotenv import load_dotenv
load_dotenv()

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores import ElasticsearchStore
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.schema import QueryBundle
from llama_index.llms import OpenAI

from .prompts import REWRITE_QUERIES_TEMPLATE, text_qa_template


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(model="gpt-3.5-turbo")


def generate_queries(query: str, num_queries: int = 3):
   response = llm.predict(
      REWRITE_QUERIES_TEMPLATE, num_queries=num_queries, query=query
   )

   queries = response.split("\n")
   queries_str = "\n".join(queries)
   print(f"Generated queries:\n{queries_str}")
   print("="*100)

   return queries

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

class ChatEngine:

    def __init__(self, documents_path="./data/", new_indexing=False):
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
        self.index = VectorStoreIndex.from_vector_store(self.vector_store, storage_context=self.storage_context)
        if new_indexing:
            self.index = VectorStoreIndex(
                self.sentence_nodes,
                storage_context=self.storage_context,
            )

    def check_update_doc(self):
        
        # Check new update document
        folder_path = './data/upload/'
        if not is_folder_empty(folder_path):
            documents = SimpleDirectoryReader(folder_path).load_data()
            
            # create the sentence window node parser w/ default settings
            node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            
            new_nodes = node_parser.get_nodes_from_documents(documents)
            self.index.insert_nodes(new_nodes)
        
            files = os.listdir(folder_path)
            for file in files: 
                source_path = f'./data/upload/{file}'
                destination_path = './data/'
                move_file(source_path, destination_path)
                
    def chat_en(self, queries: list[str], query_origin):
        self.check_update_doc()
        
        retriever = self.index.as_retriever(
            similarity_top_k=3,
            # vector_store_query_mode="hybrid",
            alpha=0.5,
            text_qa_template = text_qa_template
        )

        # Get all node after retrieval step
        retrieved_nodes = []
        for query in queries:
            retrieved_nodes += retriever.retrieve(query)
        retrieved_nodes += retriever.retrieve(query_origin)

        # Remove objects with the same content
        seen_ids = set()
        retrieved_nodes = [obj for obj in retrieved_nodes if obj.get_content() not in seen_ids and not seen_ids.add(obj.get_content())]
        print(len(retrieved_nodes))
        
        # Rerank
        query_bundle = QueryBundle(query_origin)


        rerank = SentenceTransformerRerank(
            top_n = 3,
            model = "BAAI/bge-reranker-base"
        )


        retrieved_nodes = rerank.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

        # Replace with sentence window node
        postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window",
        )

        window_nodes = postprocessor.postprocess_nodes(retrieved_nodes)
        for i in window_nodes:
            print('REFRENCES: \n')
            print(i.get_score())
            print(i.get_content())
            print('='*100)


        # Generate response with top_k result
        context_str = "\n\n".join([r.get_content() for r in window_nodes])

        llm = OpenAI(model="gpt-3.5-turbo")
        response = llm.predict(
            text_qa_template, context_str=context_str, query_str=query_origin
        )

        print(response)
        return response


# if __name__ == "__main__":
#     query = "Các cơ sở của trường đại học Tôn Đức Thắng?"
#     queries = generate_queries(query)
#     chat = ChatEngine().chat(queries, query)

#     print(chat)