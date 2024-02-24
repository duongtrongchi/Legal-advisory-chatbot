from llama_index.prompts import PromptTemplate


text_qa_template_str = (
    "Bạn là trợ lý của Andrew Huberman, người có thể đọc các ghi chú podcast của Andrew Huberman.\n"
    "Luôn trả lời truy vấn chỉ bằng cách sử dụng thông tin ngữ cảnh được cung cấp, "
    "chứ không phải kiến ​​thức sẵn có.\n"
    "Một số quy tắc cần tuân theo:\n"
    "1. Không bao giờ tham khảo trực tiếp bối cảnh nhất định trong câu trả lời của bạn.\n"
    "2. Tránh những câu như 'Dựa trên ngữ cảnh, ...' hoặc "
    "'Thông tin ngữ cảnh ...' hoặc bất cứ điều gì cùng"
    "những dòng đó."
    "Thông tin bối cảnh dưới đây.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Trả lời câu hỏi:{query_str}\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

answer_gen_str = """\
Thông tin ngữ cảnh được cung cấp dưới đây.
---------------------
{context_str}
---------------------
Sử dụng thông tin ngữ cảnh được cung cấp ở trên và không dùng kiến thức trước đó, trả lời câu truy vấn.
Câu truy vấn: {query_str}
Câu trả lời: \
"""

answer_gen_prompt = PromptTemplate(answer_gen_str)