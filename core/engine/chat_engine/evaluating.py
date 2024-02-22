import nest_asyncio

nest_asyncio.apply()


import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)
from ragas.metrics.critique import harmfulness
from ragas.llama_index import evaluate


from core.engine.chat_engine.indexing import indexing_data
from prompts import text_qa_template


if __name__ == '__main__':
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        # context_recall,
        harmfulness,
        context_relevancy,
    ]

    sentence_index = indexing_data()
    query_engine = sentence_index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
        ext_qa_template = text_qa_template
    )


    eval_questions = [
        "Tác động hạn chế cạnh tranh là gì?",
        "Thỏa thuận hạn chế cạnh tranh là gì?",
        "Thời hạn hợp đồng dầu khí kéo dài trong bao lâu?",
        "Diện tích tìm kiếm, thăm dò đối với một hợp đồng dầu khí là bao nhiêu?",
        "Khái niệm khung giá điện là gì?",
        "Các hành vi bị cấm trong việc sử dụng điện?",
        "Tổ chức sử dụng điện trong sản xuất cần có trách nhiệm gì?",
        # "Tranh chấp phát sinh giữa người tiêu dùng và tổ chức cá nhân kinh doanh hàng hóa, dịch vụ có thể giải quyết thông qua các biện pháp nào?",
        "Biên bản hòa giải phải có các nội dung chính nào?",
        "Thị trường điện lực được hình thành và phát triển theo thứ tự các cấp độ nào?",
        "Đối tượng của Thanh tra thương mại là gì?",
    ]

    eval_answers = [
        "Tác động hạn chế cạnh tranh là tác động loại trừ, làm giảm, sai lệch hoặc cản trở cạnh tranh trên thị trường.",
        "Thỏa thuận hạn chế cạnh tranh là hành vi thỏa thuận giữa các bên dưới mọi hình thức gây tác động hoặc có khả năng gây tác động hạn chế cạnh tranh.",
        "Thời hạn hợp đồng dầu khí không quá hai mươi lăm năm (25 năm), trong đó giai đoạn tìm kiếm thăm dò không quá năm năm (5 năm). Thời hạn hợp đồng dầu khí đối với khu vực nước sâu, xa bờ và thời hạn hợp đồng tìm kiếm thăm dò, khai thác khí thiên nhiên không quá ba mươi năm (30 năm), trong đó giai đoạn tìm kiếm thăm dò không quá bảy năm (7 năm).",
        "Diện tích tìm kiếm thăm dò đối với một hợp đồng dầu khí không quá hai lô (2 lô). Trong trường hợp đặc biệt, Chính phủ Việt Nam có thể cho phép diện tích tìm kiếm thăm dò đối với một hợp đồng dầu khí trên hai lô (2 lô), nhưng không quá bốn lô (4 lô).",
        "Khung giá điện là phạm vi biên độ dao động cho phép của giá điện giữa giá thấp nhất (giá sàn) và giá cao nhất (giá trần).",
        "1. Phá hoại các trang thiết bị điện, thiết bị đo đếm điện và công trình điện lực. 2. Hoạt động điện lực không có giấy phép theo quy định của Luật này. 3. Đóng, cắt điện trái quy định của pháp luật. 4. Vi phạm các quy định về an toàn trong phát điện, truyền tải điện, phân phối điện và sử dụng điện. 5. Cản trở việc kiểm tra hoạt động điện lực và sử dụng điện. 6. Trộm cắp điện. 7. Sử dụng điện để bẫy, bắt động vật hoặc làm phương tiện bảo vệ, trừ trường hợp được quy định tại Điều 59 của Luật này. 8. Vi phạm các quy định về bảo vệ hành lang an toàn lưới điện, khoảng cách an toàn của đường dây và trạm điện. 9. Cung cấp thông tin không trung thực làm tổn hại đến quyền và lợi ích hợp pháp của tổ chức, cá nhân hoạt động điện lực và sử dụng điện. 10. Lợi dụng chức vụ, quyền hạn để gây sách nhiễu, phiền hà, thu lợi bất chính trong hoạt động điện lực và sử dụng điện. 11. Các hành vi khác vi phạm quy định của pháp luật về điện lực.",
        "Tổ chức, cá nhân sử dụng điện cho sản xuất có trách nhiệm: a) Thực hiện chương trình quản lý nhu cầu điện để giảm chênh lệch công suất giữa giờ cao điểm và giờ thấp điểm của biểu đồ phụ tải hệ thống điện; b) Cải tiến, hợp lý hóa quy trình sản xuất, áp dụng công nghệ và trang thiết bị sử dụng điện có suất tiêu hao điện năng thấp để tiết kiệm điện; c) Hạn chế tối đa việc sử dụng thiết bị điện công suất lớn vào giờ cao điểm của biểu đồ phụ tải hệ thống điện; d) Bảo đảm hệ số công suất theo tiêu chuẩn kỹ thuật và hạn chế tối đa việc sử dụng non tải thiết bị điện; đ) Tổ chức kiểm toán năng lượng theo định kỳ và thực hiện các giải pháp điều chỉnh sau khi có kết luận kiểm toán theo quy định của Bộ Công nghiệp.",
        # "a) Thương lượng; b) Hòa giải; c) Trọng tài; d) Tòa án.",
        "a) Tổ chức, cá nhân tiến hành hòa giải; b) Các bên tham gia hòa giải; c) Nội dung hòa giải; d) Thời gian, địa điểm tiến hành hòa giải; đ) Ý kiến của các bên tham gia hòa giải; e) Kết quả hòa giải; g) Thời hạn thực hiện kết quả hòa giải thành.",
        "Thị trường điện lực được hình thành và phát triển theo thứ tự các cấp độ sau đây: a) Thị trường phát điện cạnh tranh; b) Thị trường bán buôn điện cạnh tranh; c) Thị trường bán lẻ điện cạnh tranh.",
        "Đối tượng của Thanh tra thương mại là hoạt động thương mại của thương nhân.",

    ]

    eval_answers = [[a] for a in eval_answers]


    result = evaluate(query_engine, metrics, eval_questions, eval_answers)
    print(result)