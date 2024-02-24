from llama_index.indices.postprocessor import MetadataReplacementPostProcessor


from indexing import indexing_data
from prompts import text_qa_template

# Cài thư viện
# !pip install torch sentence-transformers
from llama_index.postprocessor import SentenceTransformerRerank

if __name__ == '__main__':
    # Define reranker model
    rerank_model = SentenceTransformerRerank(
        top_n = 2, 
        model = "BAAI/bge-reranker-base"
    )
    
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

    window_response = query_engine.query(
        # "Nghĩa vụ của thương nhân khi thanh tra thương mại thực hiện việc kiễm tra là gì?"
        "Trộm cắp điện là hành vi gì?",
        # "Bệnh lậu là bệnh gì?"
    )
    print(window_response)


    # for i in window_response.source_nodes:
    #     print('REFRENCES: \n')
    #     print(i.score)
    #     print(i.text)
    #     print('='*100)
