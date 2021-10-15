from django.shortcuts import render
from django.http import HttpResponse

def signup(request):
    return render(request,'signup.html')


def login(request):
    return render(request,'login.html')

# Create your views here.
