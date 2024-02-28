import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

from datasets import Dataset

import openai

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
import openai


os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')


from prompts import text_qa_template
from simpleRAG import genaration_qa


data_list = [
    {
        'question': "Hoạt động điện lực là gì?",
        'ground_truth': """Hoạt động điện lực là hoạt động của tổ chức, cá nhân trong các lĩnh vực quy hoạch, đầu tư phát triển điện lực, phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan.""",
        'answer': "Hoạt động điện lực là hoạt động của tổ chức, cá nhân trong các lĩnh vực quy hoạch, đầu tư phát triển điện lực, phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan. Đơn vị điện lực là tổ chức, cá nhân thực hiện các hoạt động này.",
        'contexts': ["Trường hợp điều ước quốc tế mà nước Cộng hòa xã hội chủ nghĩa Việt Nam ký kết hoặc gia nhập có quy định khác với quy định của Luật này thì áp dụng quy định của điều ước quốc tế đó.  Điều 3.  Giải thích từ ngữ Trong Luật này, các từ ngữ dưới đây được hiểu như sau: 1.  Hoạt động điện lực là hoạt động của tổ chức, cá nhân trong các lĩnh vực quy hoạch, đầu tư phát triển điện lực, phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan.  2.  Đơn vị điện lực là tổ chức, cá nhân thực hiện hoạt động phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan."]
    },
    {
        'question': "Thị trường điện lực được hình thành và phát triển dựa vào các cấp độ nào?",
        'ground_truth': "1. Thị trường điện lực được hình thành và phát triển theo thứ tự các cấp độ sau đây: a) Thị trường phát điện cạnh tranh; b) Thị trường bán buôn điện cạnh tranh; c) Thị trường bán lẻ điện cạnh tranh.",
        'answer': "Thị trường điện lực được hình thành và phát triển dựa vào ba cấp độ chính, đó là thị trường phát điện cạnh tranh, thị trường bán buôn điện cạnh tranh và thị trường bán lẻ điện cạnh tranh. Nhà nước điều tiết hoạt động của thị trường điện lực nhằm bảo đảm phát triển hệ thống điện bền vững, đáp ứng yêu cầu cung cấp điện an toàn, ổn định, hiệu quả.",
        'contexts': ["Nhà nước điều tiết hoạt động của thị trường điện lực nhằm bảo đảm phát triển hệ thống điện bền vững, đáp ứng yêu cầu cung cấp điện an toàn, ổn định, hiệu quả.  Điều 18.  Hình thành và phát triển thị trường điện lực  1.  Thị trường điện lực được hình thành và phát triển theo thứ tự các cấp độ sau đây: a) Thị trường phát điện cạnh tranh;  b) Thị trường bán buôn điện cạnh tranh;  c) Thị trường bán lẻ điện cạnh tranh.   2.  Thủ tướng Chính phủ quy định lộ trình, các điều kiện để hình thành và phát triển các cấp độ thị trường điện lực. ", "3.  Nhà nước điều tiết hoạt động của thị trường điện lực nhằm bảo đảm phát triển hệ thống điện bền vững, đáp ứng yêu cầu cung cấp điện an toàn, ổn định, hiệu quả.  Điều 18.  Hình thành và phát triển thị trường điện lực  1.  Thị trường điện lực được hình thành và phát triển theo thứ tự các cấp độ sau đây: a) Thị trường phát điện cạnh tranh;  b) Thị trường bán buôn điện cạnh tranh;  c) Thị trường bán lẻ điện cạnh tranh.   2. ", "2.  Tôn trọng quyền được tự chọn đối tác và hình thức giao dịch của các đối tượng mua bán điện trên thị trường phù hợp với cấp độ phát triển của thị trường điện lực.  3.  Nhà nước điều tiết hoạt động của thị trường điện lực nhằm bảo đảm phát triển hệ thống điện bền vững, đáp ứng yêu cầu cung cấp điện an toàn, ổn định, hiệu quả.  Điều 18.  Hình thành và phát triển thị trường điện lực  1. "]
    },
    {
        'question': "Hành lang bảo vệ an toàn lưới điện cao áp bao gồm?",
        'ground_truth': "2. Hành lang bảo vệ an toàn lưới điện cao áp bao gồm: a) Hành lang bảo vệ an toàn đường dây dẫn điện trên không; b) Hành lang bảo vệ an toàn đường cáp điện ngầm; c) Hành lang bảo vệ an toàn trạm điện. 3. Chính phủ quy định cụ thể về hành lang bảo vệ an toàn lưới điện cao áp.",
        'answer': "Hành lang bảo vệ an toàn lưới điện cao áp bao gồm hành lang bảo vệ an toàn đường dây dẫn điện trên không, hành lang bảo vệ an toàn đường cáp điện ngầm và hành lang bảo vệ an toàn trạm điện.",
        'contexts': ["Hành lang bảo vệ an toàn lưới điện cao áp  1.  Hành lang an toàn lưới điện cao áp là khoảng không gian giới hạn dọc theo đường dây tải điện hoặc bao quanh trạm điện và được quy định cụ thể theo từng cấp điện áp.  2.  Hành lang bảo vệ an toàn lưới điện cao áp bao gồm: a) Hành lang bảo vệ an toàn đường dây dẫn điện trên không;  b) Hành lang bảo vệ an toàn đường cáp điện ngầm;  c) Hành lang bảo vệ an toàn trạm điện.  3.  Chính phủ quy định cụ thể về hành lang bảo vệ an toàn lưới điện cao áp. ", "3.  Trường hợp các bên liên quan không thoả thuận được thì yêu cầu cơ quan nhà nước có thẩm quyền giải quyết và triển khai thực hiện theo quyết định của cơ quan nhà nước có thẩm quyền.  Điều 50.  Hành lang bảo vệ an toàn lưới điện cao áp  1.  Hành lang an toàn lưới điện cao áp là khoảng không gian giới hạn dọc theo đường dây tải điện hoặc bao quanh trạm điện và được quy định cụ thể theo từng cấp điện áp.  2. ", "5. Hệ thống cáp điện trong nhà máy điện, trạm phát điện phải đáp ứng các quy định về an toàn sau đây: a) Cáp điện phải được sắp xếp trật tự theo chủng loại, tính năng kỹ thuật, cấp điện áp và được đặt trên các giá đỡ. Cáp điện đi qua khu vực có ảnh hưởng của nhiệt độ cao phải được cách nhiệt và đi trong ống bảo vệ; b) Hầm cáp, mương cáp phải có nắp đậy kín, thoát nước tốt, bảo quản sạch sẽ, khô ráo. Không được để nước, dầu, hoá chất, tạp vật tích tụ trong hầm cáp, mương cáp. Hầm cáp phải có tường ngăn để tránh hỏa hoạn lan rộng; có hệ thống báo cháy và chữa cháy tự động, hệ thống đèn chiếu sáng sử dụng điện áp an toàn phù hợp với quy phạm, tiêu chuẩn kỹ thuật an toàn điện. 6. Các trang thiết bị và hệ thống chống sét, nối đất trong nhà máy điện, trạm phát điện, trạm phân phối điện phải được lắp đặt đúng thiết kế và được kiểm tra nghiệm thu, kiểm tra định kỳ theo đúng quy phạm, tiêu chuẩn kỹ thuật an toàn điện. Điều 55. An toàn trong truyền tải điện, phân phối điện 1. Chủ công trình lưới điện phải chịu trách nhiệm: a) Đặt biển báo an toàn về điện tại các trạm điện, cột điện; b) Sơn màu và đặt đèn tín hiệu trên đỉnh cột tại các cột có độ cao và vị trí đặc biệt để bảo vệ an toàn lưới điện cao áp. 2. ở các vị trí giao chéo giữa đường dây dẫn điện cao áp trên không, đường cáp điện ngầm với đường sắt, đường bộ, đường thuỷ nội địa, việc đặt và quản lý biển báo, biển cấm vượt qua đối với phương tiện vận tải được thực hiện theo quy định của Bộ Giao thông vận tải. Chủ đầu tư công trình xây dựng sau phải chịu chi phí cho việc đặt biển báo, biển cấm. 3. Khi bàn giao công trình lưới điện, chủ đầu tư công trình phải giao cho đơn vị quản lý vận hành lưới điện đầy đủ các tài liệu kỹ thuật, biên bản nghiệm thu, quyết định giao đất, cho thuê đất và các tài liệu liên quan đến đền bù, giải phóng mặt bằng theo quy định của pháp luật. 4."]
    },
    {
        'question': "Thế nào là doanh nghiệp được coi là có vị trí độc quyền?",
        'ground_truth': """Doanh nghiệp được coi là có vị trí độc quyền nếu không có doanh nghiệp nào cạnh tranh về hàng hóa, dịch vụ mà doanh nghiệp đó kinh doanh trên thị trường liên quan.""",
        'answer': "Doanh nghiệp được coi là có vị trí độc quyền khi không có doanh nghiệp nào cạnh tranh về hàng hóa, dịch vụ mà doanh nghiệp đó kinh doanh trên thị trường liên quan. Điều này có thể dẫn đến việc lạm dụng vị trí thống lĩnh thị trường và hành vi cạnh tranh không lành mạnh, gây thiệt hại đến quyền và lợi ích hợp pháp của doanh nghiệp khác.",
        'contexts': ["3.  Nhóm doanh nghiệp có vị trí thống lĩnh thị trường quy định tại khoản 2 Điều này không bao gồm doanh nghiệp có thị phần ít hơn 10% trên thị trường liên quan.   Điều 25.  Doanh nghiệp có vị trí độc quyền Doanh nghiệp được coi là có vị trí độc quyền nếu không có doanh nghiệp nào cạnh tranh về hàng hóa, dịch vụ mà doanh nghiệp đó kinh doanh trên thị trường liên quan.   Điều 26.  Xác định sức mạnh thị trường đáng kể 1.",
                     "25.  Lạm dụng vị trí thống lĩnh thị trường, lạm dụng vị trí độc quyền là hành vi của doanh nghiệp có vị trí thống lĩnh thị trường, vị trí độc quyền gây tác động hoặc có khả năng gây tác động hạn chế cạnh tranh.  6.  Hành vi cạnh tranh không lành mạnh là hành vi của doanh nghiệp trái với nguyên tắc thiện chí, trung thực, tập quán thương mại và các chuẩn mực khác trong kinh doanh, gây thiệt hại hoặc có thể gây thiệt hại đến quyền và lợi ích hợp pháp của doanh nghiệp khác.",
                     "Doanh nghiệp có vị trí độc quyền Doanh nghiệp được coi là có vị trí độc quyền nếu không có doanh nghiệp nào cạnh tranh về hàng hóa, dịch vụ mà doanh nghiệp đó kinh doanh trên thị trường liên quan.   Điều 26.  Xác định sức mạnh thị trường đáng kể 1.  Sức mạnh thị trường đáng kể của doanh nghiệp, nhóm doanh nghiệp được xác định căn cứ vào một số yếu tố sau đây:  a) Tương quan thị phần giữa các doanh nghiệp trên thị trường liên quan; b) Sức mạnh tài chính, quy mô của doanh nghiệp; c) Rào cản gia nhập, mở rộng thị trường đối với doanh nghiệp khác; d) Khả năng nắm giữ, tiếp cận, kiểm soát thị trường phân phối, tiêu thụ hàng hóa, dịch vụ hoặc nguồn cung hàng hóa, dịch vụ; đ) Lợi thế về công nghệ, hạ tầng kỹ thuật; e) Quyền sở hữu, nắm giữ, tiếp cận cơ sở hạ tầng; g) Quyền sở hữu, quyền sử dụng đối tượng quyền sở hữu trí tuệ; h) Khả năng chuyển sang nguồn cung hoặc cầu đối với các hàng hóa, dịch vụ liên quan khác;"]
    },
    {
        'question': "Khí thiên nhiên là khí gì?",
        'ground_truth': """"Khí thiên nhiên" là toàn bộ hydrocarbon ở thể khí, khai thác từ giếng khoan, bao gồm cả khí ẩm, khí khô, khí đầu giếng khoan và khí còn lại sau khi chiết xuất hydrocarbon lỏng từ khí ẩm.""",
        'answer': "Khí thiên nhiên là toàn bộ hydrocarbon ở thể khí, bao gồm cả khí ẩm, khí khô, khí đầu giếng khoan và khí còn lại sau khi chiết xuất hydrocarbon lỏng từ khí ẩm.",
        'contexts': ["""2.  "Dầu thô" là hydrocarbon ở thể lỏng trong trạng thái tự nhiên, asphalt, ozokerite và hydrocarbon lỏng thu được từ khí thiên nhiên bằng phương pháp ngưng tụ hoặc chiết xuất.  3.  "Khí thiên nhiên" là toàn bộ hydrocarbon ở thể khí, khai thác từ giếng khoan, bao gồm cả khí ẩm, khí khô, khí đầu giếng khoan và khí còn lại sau khi chiết xuất hydrocarbon lỏng từ khí ẩm.  4.  "Hoạt động dầu khí" là hoạt động tìm kiếm thăm dò, phát triển mỏ và khai thác dầu khí, kể cả các hoạt động phục vụ trực tiếp cho các hoạt động này.""",
                     """Điều 2 Nhà nước Việt Nam khuyến khích các tổ chức, cá nhân Việt Nam và nước ngoài đầu tư vốn, công nghệ để tiến hành các hoạt động dầu khí trên cơ sở tôn trọng độc lập, chủ quyền, toàn vẹn lãnh thổ, an ninh quốc gia của Việt Nam và tuân thủ pháp luật Việt Nam.  Nhà nước Việt Nam bảo hộ quyền sở hữu đối với vốn đầu tư, tài sản và các quyền lợi hợp pháp khác của các tổ chức, cá nhân Việt Nam và nước ngoài tiến hành các hoạt động dầu khí ở Việt Nam.  Điều 3 Trong Luật này, các từ ngữ dưới đây được hiểu như sau: 1.  "Dầu khí" là dầu thô, khí thiên nhiên và hydrocarbon ở thể khí, lỏng, rắn hoặc nửa rắn trong trạng thái tự nhiên, kể cả sulphur và các chất tương tự khác kèm theo hydrocarbon nhưng không kể than, đá phiến sét, bitum hoặc các khoáng sản khác có thể chiết xuất được dầu.  2.  "Dầu thô" là hydrocarbon ở thể lỏng trong trạng thái tự nhiên, asphalt, ozokerite và hydrocarbon lỏng thu được từ khí thiên nhiên bằng phương pháp ngưng tụ hoặc chiết xuất.""",
                     """10.  "Xí nghiệp liên doanh dầu khí" là Xí nghiệp liên doanh được thành lập trên cơ sở hợp đồng dầu khí hoặc trên cơ sở Hiệp định ký kết giữa Chính phủ Việt Nam với Chính phủ nước ngoài.    CHƯƠNG II HOẠT ĐỘNG DẦU KHÍ Điều 4 Tổ chức, cá nhân tiến hành hoạt động dầu khí phải sử dụng kỹ thuật, công nghệ tiên tiến, tuân thủ các quy định của pháp luật Việt Nam về bảo vệ tài nguyên, bảo vệ môi trường, an toàn cho người và tài sản.  Điều 5 Tổ chức, cá nhân tiến hành hoạt động dầu khí phải có đề án bảo vệ môi trường, thực hiện tất cả các biện pháp để ngăn ngừa ô nhiễm, loại trừ ngay các nguyên nhân gây ra ô nhiễm và có trách nhiệm khắc phục hậu quả do sự cố ô nhiễm môi trường gây ra.  Điều 6 Tổ chức, cá nhân tiến hành hoạt động dầu khí phải thiết lập vùng an toàn cho các công trình phục vụ hoạt động dầu khí theo quy định của Chính phủ Việt Nam.    Điều 7 Tổ chức, cá nhân tiến hành hoạt động dầu khí phải mua bảo hiểm đối với các phương tiện, công trình phục vụ hoạt động dầu khí, bảo hiểm môi trường và các bảo hiểm khác theo quy định của pháp luật Việt Nam, phù hợp với thông lệ trong công nghiệp dầu khí quốc tế."""]
    }
]
ds = Dataset.from_list(data_list)



if __name__ == "__main__":
    result = evaluate(
        ds,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    print(result)

