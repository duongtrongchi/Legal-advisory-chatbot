from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms import OpenAI
from llama_index.schema import QueryBundle

from indexing import indexing_data
from prompts import text_qa_template, answer_gen_prompt
from rewrite import generate_queries

# Cài thư viện
# !pip install torch sentence-transformers
from llama_index.postprocessor import SentenceTransformerRerank


def generate_answer(queries: list[str], query_origin):
    
    sentence_index = indexing_data()
    
    # Retriever engine
    retriever = sentence_index.as_retriever(
        similarity_top_k=3,
        vector_store_query_mode="hybrid", 
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
    print(retrieved_nodes)
    
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
        answer_gen_prompt, context_str=context_str, query_str=query_origin
    )
    
    print(response)
    return response

if __name__ == '__main__':
    query = "Trường TDTU có phải là trường nằm trong TOP hay không?"
    queries = generate_queries(query)
    generate_answer(queries, query)
