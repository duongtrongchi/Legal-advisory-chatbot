from dotenv import load_dotenv
load_dotenv()

import json

from django.shortcuts import render


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from .chat_engine.chat import ChatEngine, generate_queries
# from .chat_engine.intent import intent_classification
from .chat_engine.testIntent import intent_classification



class ChatEngineView(APIView):

    def post(self, request ,format=None):
        question = request.data.get('question', None)

        if question is None:
            return Response({'response': 'Không nhận được câu hỏi!!!'}, status=status.HTTP_400_BAD_REQUEST)

        intent = json.loads(intent_classification(question))
        print("INTENT:")
        print(intent)
        
        if intent['response'] == "Chào hỏi":
            return Response({'response': "Xin chào, tôi là trợ lý ảo có thể trả lời các câu hỏi về pháp luật, tôi có thể giúp gì cho bạn?"}, status=status.HTTP_200_OK)
        elif intent['response'] == "Chủ đề khác":
            return Response({'response': "Câu hỏi của bạn không hợp lệ, hoặc không liên quan đến Pháp Luật. Vui lòng kiểm tra lại và cung cấp thêm thông tin cho tôi"}, status=status.HTTP_200_OK)
        else:
            queries = generate_queries(question)
            response = ChatEngine()
            response, references = response.chat_en(queries, question)
            
            print("references:")
            print(references)
            print("="*100)
            return Response({'response': response, 'references': references}, status=status.HTTP_200_OK)
    # def post(self, request ,format=None):
    #     question = request.data.get('question', None)

    #     if question is None:
    #         return Response({'response': 'Không nhận được câu hỏi!!!'}, status=status.HTTP_400_BAD_REQUEST)

    #     queries = generate_queries(question)
    #     response = ChatEngine()
    #     response, references = response.chat_en(queries, question)

    #     return Response({'response': response}, status=status.HTTP_200_OK)


def get_template(request):
    if request.method == 'POST':
        question = request.data.get('question', None)

        if question is None:
            return render(request, 'engine/component.html', {'response':"Không nhận được câu trả lời!!!"})

        queries = generate_queries(question)
        response = ChatEngine()
        response = response.chat_en(queries, question)
        return render(request, 'engine/component.html', {'response':response})

    return render(request, 'engine/component.html', {'response':"ERROR!!!"})