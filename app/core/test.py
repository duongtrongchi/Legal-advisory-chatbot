from llama_index.prompts import PromptTemplate
from llama_index.llms import OpenAI
from indexing import indexing_data
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from prompts import text_qa_template
from llama_index.postprocessor import SentenceTransformerRerank

query_gen_str = """\
Bạn là một trợ lý hữu ích trong việc tạo ra nhiều câu truy vấn liên quan nhất dựa trên \
một câu truy vấn đầu vào. Tạo {num_queries} câu truy vấn, một câu trên mỗi dòng, sắp xếp thứ tự theo độ liên quan từ cao xuống thấp,\
liên quan tới câu truy vấn dưới đây:
Câu truy vấn: {query}
Các câu truy vấn tạo ra:
"""

answer_gen_str = """\
Bạn là một trợ lý hữu ích trong việc tổng hợp các câu trả lời từ một danh sách \
các câu trả lời đầu vào và trả lời cho một câu truy vấn từ người dùng. Tạo câu trả lời từ danh sách các câu trả lời và câu truy vấn dưới đây: 
Danh sách câu trả lời: {answers}
Câu truy vấn: {query}
Câu trả lời tạo ra:
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
    # assume LLM proper put each query on a newline
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    print("="*100)
    return queries

def generate_answers(queries: list[str], llm, query_origin):
    sentence_index = indexing_data()
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            rerank_model,
        ],
        vector_store_query_mode="hybrid", 
        alpha=0.5,
        text_qa_template = text_qa_template
    )
    
    answers = []
    for query in queries:
        window_response = query_engine.query(query)
        answers.append(window_response.source_nodes[0].text)
        answers.append(window_response.source_nodes[1].text)
    
    print(answers)
    response = llm.predict(
        answer_gen_prompt, answers=answers, query=query_origin
    )
    
    answer = response
    print(f"Generated answer:\n{answer}")
    print("="*100)
    return answer

query = "Tôi muốn đăng ký kinh doanh thì tôi cần làm gì?"
queries = generate_queries(query, llm)
generate_answers(queries, llm, query)