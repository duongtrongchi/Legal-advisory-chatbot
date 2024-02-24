from llama_index.prompts import PromptTemplate
from llama_index.llms import OpenAI
from indexing import indexing_data
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from prompts import text_qa_template
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.schema import QueryBundle

query_gen_str = """\
Bạn là một trợ lý hữu ích trong việc tạo ra nhiều câu truy vấn liên quan nhất dựa trên \
một câu truy vấn đầu vào. Tạo {num_queries} câu truy vấn, một câu trên mỗi dòng, sắp xếp thứ tự theo độ liên quan từ cao xuống thấp,\
liên quan tới câu truy vấn dưới đây:
Câu truy vấn: {query}
Các câu truy vấn tạo ra:
"""

answer_gen_str = """\
Thông tin ngữ cảnh được cung cấp dưới đây.
---------------------
{context_str}
---------------------
Sử dụng thông tin ngữ cảnh được cung cấp và không dùng kiến thức trước đó, trả lời câu truy vấn.
Câu truy vấn: {query_str}
Câu trả lời: \
"""

query_gen_prompt = PromptTemplate(query_gen_str)
answer_gen_prompt = PromptTemplate(answer_gen_str)

llm = OpenAI(model="gpt-3.5-turbo")

rerank_model = SentenceTransformerRerank(
    top_n = 2, 
    model = "BAAI/bge-reranker-base"
)

def generate_queries(query: str, llm, num_queries: int = 3):
   
   response = llm.predict(
      query_gen_prompt, num_queries=num_queries, query=query
   )
    
   queries = response.split("\n")
   queries_str = "\n".join(queries)
   print(f"Generated queries:\n{queries_str}")
   print("="*100)
   return queries

def generate_answers(queries: list[str], llm, query_origin):
   
   sentence_index = indexing_data()
   retriever = sentence_index.as_retriever(
      similarity_top_k=3,
      vector_store_query_mode="hybrid", 
      alpha=0.5,
      text_qa_template = text_qa_template
   )

   retrieved_nodes = []
   for query in queries:
      retrieved_nodes += retriever.retrieve(query)
   
   rerank = SentenceTransformerRerank(
      top_n = 2, 
      model = "BAAI/bge-reranker-base"
   )
   
   query_bundle = QueryBundle(query_origin)
   retrieved_nodes = rerank.postprocess_nodes(
      retrieved_nodes, query_bundle
   )
   
   postprocessor = MetadataReplacementPostProcessor(
      target_metadata_key="window",
   )
   window_nodes = postprocessor.postprocess_nodes(retrieved_nodes)
   
   print(window_nodes)
   for i in window_nodes:
        print('REFRENCES: \n')
        print(i.get_score())
        print(i.get_content())
        print('='*100)
    
   context_str = "\n\n".join([r.get_content() for r in window_nodes])
   response = llm.predict(
      answer_gen_prompt, context_str=context_str, query_str=query_origin
   )
   
   print(response)
   return response

query = "Tôi muốn đăng ký kinh doanh thì tôi cần làm gì?"
queries = generate_queries(query, llm)
generate_answers(queries, llm, query)