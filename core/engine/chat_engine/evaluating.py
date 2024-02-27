import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness



os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')


from prompts import text_qa_template
from simpleRAG import genaration_qa


data_list = [
    {
        'question': "Hoạt động điện lực là gì?",
        'ground_truth': """Hoạt động điện lực là hoạt động của tổ chức, cá nhân trong các lĩnh vực quy hoạch, đầu tư phát triển điện lực, phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan.""",
        'answer': """Hoạt động điện lực là hoạt động của tổ chức, cá nhân trong các lĩnh vực quy hoạch, đầu tư phát triển điện lực, phát điện, truyền tải điện, phân phối điện, điều độ hệ thống điện, điều hành giao dịch thị trường điện lực, bán buôn điện, bán lẻ điện, tư vấn chuyên ngành điện lực và những hoạt động khác có liên quan. Đơn vị điện lực là tổ chức, cá nhân thực hiện các hoạt động này.""",
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

