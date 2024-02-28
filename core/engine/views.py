from dotenv import load_dotenv
load_dotenv()

import json

from django.shortcuts import render


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from .chat_engine.chat import ChatEngine, generate_queries
from .chat_engine.intent import intent_classification



class ChatEngineView(APIView):

    def post(self, request ,format=None):
        question = request.data.get('question', None)

        if question is None:
            return Response({'response': 'Không nhận được câu hỏi!!!'}, status=status.HTTP_400_BAD_REQUEST)

        intent = json.loads(intent_classification(question))
        if intent['response'] == 1:
            print("INTENT:")
            print(intent)
            return Response({'response': "Xin chào"}, status=status.HTTP_200_OK)
        else:
            print("INTENT:")
            print(intent)
            queries = generate_queries(question)
            response = ChatEngine()
            response = response.chat_en(queries, question)
            return Response({'response': response}, status=status.HTTP_200_OK)


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