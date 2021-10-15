from django.urls import path
from . import views

urlpatterns = [
    path('', views.signup, name='signup'),
    path('/returning_user', views.login, name='returning_user')
]