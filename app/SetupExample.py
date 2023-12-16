from dotenv import load_dotenv

import openai
import os
load_dotenv()


openai.api_key = os.getenv('OPENAI_API_KEY')


def multy_choices(context):

    prompt = f"""
    context: {context}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """Bạn là một giảng viên đại học chuyên nghành về sức khoẻ. \
            Nhiệm vụ của bạn là dựa vào context được cung cấp sẵn hãy tạo ra một bộ câu hỏi trắc nghiệm cho sinh viên để giúp họ kiểm tra kiến thức trong kì thi sắp tới. \
            Bộ câu hỏi trắc nghiệm của bạn bao gồm câu hỏi và câu các lựa chọn: A, B, C, D và trong các lựa chọn này phải có một lựa chọn đúng. \

            Định dạng trả lời là một file json trong python với các thuộc tính: "question", "options", "answer".\
            Trong đó, "question" là câu hỏi mà bạn đặt ra cho sinh viên, "options" là các lựa chọn A, B, C, D dựa vào context, "answer" là đáp án đúng từ các "options"
            Bạn nên dựa vào ví dụ sau đây mà tôi cung cấp để đưa ra được định dạng trả lời chính xác.\

             Ví dụ:\
             [
             {"question": "...", "options": [{"answer": "...", "key": "A"}, {"answer": "...", "key": "B"}, {"answer": "...", "key": "C"}, {"answer": "...", "key": "D"}], "answer": "..."},
             {"question": "...", "options": [{"answer": "...", "key": "A"}, {"answer": "...", "key": "B"}, {"answer": "...", "key": "C"}, {"answer": "...", "key": "D"}], "answer": "..."},
             ]"""},
        {"role": "user", "content": prompt}
        ],
        temperature=0
    )


    return response['choices'][0]['message']['content']



context = """
Loét Buruli (cũng còn gọi là Loét Bairnsdale, Loét Searls, hay Loét Daintree) là bệnh truyền nhiễm do vi khuẩn Mycobacterium ulcerans gây ra. Biểu hiện đặc trưng ở giai đoạn đầu của bệnh là một u nhỏ hoặc một vùng bị sưng. U nhỏ có thể chuyển thành loét. Loét có thể rộng ở bên trong hơn so với ở bề mặt da, và sưng ở xung quanh. Khi bệnh nặng hơn có thể ảnh hưởng đến xương. Bệnh loét Buruli thường rảy ra nhất ở tay hoặc chân; ít khi có sốt.
"""
res = multy_choices(context)
print(res)