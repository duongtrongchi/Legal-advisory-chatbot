from django.urls import path
from .views import ChatEngine


urlpatterns = [
    path('chat/', ChatEngine.as_view(), name="chat-engine"),
]