from llama_index.indices.postprocessor import MetadataReplacementPostProcessor


from indexing import indexing_data
from prompts import text_qa_template


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
        "Đơn vị điện lực là gì?"
    )
    print(window_response)