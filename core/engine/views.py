from dotenv import load_dotenv
load_dotenv()

from django.shortcuts import render


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from .chat_engine.chat import ChatEngine, generate_queries


class ChatEngineView(APIView):

    def post(self, request ,format=None):
        question = request.data.get('question', None)

        if question is None:
            return Response({'response': 'Không nhận được câu hỏi!!!'}, status=status.HTTP_400_BAD_REQUEST)

        queries = generate_queries(question)
        response = ChatEngine()
        response = response.chat_en(queries, question)

        return Response({'response': response}, status=status.HTTP_200_OK)


def get_template(request):
    return render(request, 'engine/component.html', {'project_list':"Hello world"})