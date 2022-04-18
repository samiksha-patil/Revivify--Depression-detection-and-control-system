from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
# Create your views here.
from .forms import UserRegisterForm
import matplotlib.pyplot as plt
import numpy as np

# Create your views here.
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'{ username }, Your account has been created! You are now able to login')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request,'users/register.html',{'form': form})

def home(request):
     
     return render(request,'home.html')

def twitter(request):
    Signal_1 = np.random.randint(low = 0, high = 50, size = 20)
    Signal_2 = np.random.randint(low = 0, high = 10, size = 20)
    Signal_3 = np.random.randint(low = 20, high = 30, size = 20)
    Signal_4 = np.random.randint(low = 10, high = 30, size = 20)
    Signal_5 = np.random.randint(low = 0, high = 1, size = 20)
    Signal_6 = np.random.randint(low = 0, high = 90, size = 20)
    Signal_7 = np.random.randint(low = 0, high = 40, size = 20)
    Signal_8 = np.random.randint(low = 10, high = 60, size = 20)
    Signal_9 = np.random.randint(low = 0, high = 10, size = 20)

    fig, ax = plt.subplots()

    ax.plot(Signal_1, color = 'green', label = 'Signal_1')
    ax.plot(Signal_2, color = 'blue', label = 'Signal_2')
    ax.plot(Signal_3, color = 'orange', label = 'Signal_3')
    ax.plot(Signal_4, color = 'gray', label = 'Signal_4')
    ax.plot(Signal_5, color = 'skyblue', label = 'Signal_5')
    ax.plot(Signal_6, color = 'brown', label = 'Signal_6')
    ax.plot(Signal_7, color = 'plum', label = 'Signal_7')
    ax.plot(Signal_8, color = 'palegreen', label = 'Signal_8')
    ax.plot(Signal_9, color = 'lightcoral', label = 'Signal_9')
    ax.legend(loc = 'upper right')
    plt.savefig("users/static/image/twitter.png")
    #plt.show() 

    return render(request,'twitter.html')

# def login(request):
#     return render(request,'users/login.html')