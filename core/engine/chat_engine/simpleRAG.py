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
    es_url="http://localhost:9200",
    index_name="simple_rag",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context,
)


query_engine = index.as_query_engine(text_qa_template=base_prompt_template)
response = query_engine.query("Cho tôi biết đại học Tôn Đức Thắng có bao nhiêu cơ sở chính và đó là cơ sở nào?")
print(response)