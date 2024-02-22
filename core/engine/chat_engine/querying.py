from llama_index.indices.postprocessor import MetadataReplacementPostProcessor


from core.engine.chat_engine.indexing import indexing_data
from core.engine.chat_engine.prompts import text_qa_template


if __name__ == '__main__':
    sentence_index = indexing_data()
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
        ext_qa_template = text_qa_template
    )


    window_response = query_engine.query(
        "Nghĩa vụ của thương nhân khi thanh tra thương mại thực hiện việc kiễm tra là gì?"
    )
    print(window_response)


    for i in window_response.source_nodes:
        print('REFRENCES: \n')
        print(i.score)
        print(i.text)
        print('='*100)
