import logging
import sys
import os

import tiktoken
import openai
from dotenv import load_dotenv
load_dotenv()


from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ElasticsearchStore
from llama_index.storage.storage_context import StorageContext


from prompts import base_prompt_template

documents = SimpleDirectoryReader("../../data/").load_data()


node_parser = SimpleNodeParser.from_defaults(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

llm = OpenAI(model='gpt-3.5-turbo', temperature=0.7, max_tokens=256)
embed_model = OpenAIEmbedding()

prompt_helper = PromptHelper(
  context_window=4096,
  num_output=256,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)


vector_store = ElasticsearchStore(
  index_name="law_bot",
  es_cloud_id="a360a60c18784a4288ef610006c3b861:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDAwZGM4M2JiYjU3NjRjZTliZDJlYjEyNTAwNTA2N2MxJDQzOTI5MzIyNGNlMjRiZDZhOTRkODYzOWQyZTNlYWJl",
  es_api_key="bUR1b3lZMEIzSUxOY1MxYjRvMEQ6ZE9PMS01UGlSSVdvdEhncUVkWmlWQQ=="
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)


def indexing_simple_rag(flag=False, path="../../data/"):
    index = None
    if flag == True:
        documents = SimpleDirectoryReader(path).load_data()
        index = VectorStoreIndex.from_documents(
          documents,
          service_context=service_context,
          storage_context=storage_context,
        )
        return index

    index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            service_context=service_context
    )
    return index


def genaration_qa(question, new_index=False, path="../../data/"):
  if new_index == True:
    index = indexing_simple_rag(flag=new_index, path=path)
  else:
     index = indexing_simple_rag(flag=False)
  query_engine = index.as_query_engine(text_qa_template=base_prompt_template)
  response = query_engine.query(question)
  return response


if __name__ == "__main__":
   ques = genaration_qa(question="Bạn là ai?")
   print(ques)