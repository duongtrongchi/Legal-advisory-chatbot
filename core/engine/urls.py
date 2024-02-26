from django.urls import path
from .views import ChatEngineView


urlpatterns = [
    path('chat/', ChatEngineView.as_view(), name="chat-engine"),
]