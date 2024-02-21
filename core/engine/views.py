from dotenv import load_dotenv
load_dotenv()

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


from .chat_engine.chat import generation


class ChatEngine(APIView):

    def post(self, request ,format=None):
        question = request.data.get('question', None)

        if question is None:
            return Response({'response': 'Không nhận được câu hỏi!!!'}, status=status.HTTP_400_BAD_REQUEST)

        response = generation(question)
        return Response({'response': response}, status=status.HTTP_200_OK)