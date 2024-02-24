from llama_index.prompts import PromptTemplate
from llama_index.llms import OpenAI

query_gen_str = """\
Bạn là một trợ lý hữu ích trong việc tạo ra nhiều câu truy vấn tương tự hoặc có liên quan nhất dựa trên \
một câu truy vấn đầu vào. Tạo {num_queries} câu truy vấn, một câu trên mỗi dòng, sắp xếp thứ tự theo độ liên quan từ cao xuống thấp,\
liên quan tới câu truy vấn dưới đây:
Câu truy vấn: {query}
Các câu truy vấn tạo ra:
"""

query_gen_prompt = PromptTemplate(query_gen_str)

llm = OpenAI(model="gpt-3.5-turbo")

def generate_queries(query: str, num_queries: int = 3):
   response = llm.predict(
      query_gen_prompt, num_queries=num_queries, query=query
   )
    
   queries = response.split("\n")
   queries_str = "\n".join(queries)
   print(f"Generated queries:\n{queries_str}")
   print("="*100)
   
   return queries
