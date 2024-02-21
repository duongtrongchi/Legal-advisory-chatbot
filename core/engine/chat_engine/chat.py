import openai
import os


os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv('OPENAI_API_KEY')

from openai import OpenAI
client = OpenAI()


def generation(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """Bạn là một giảng viên đại học nhiệm vụ của bạn là khi bạn nhận được một câu hỏi hãy trả lời câu hỏi mà người dùng yêu cầu"""},

            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content