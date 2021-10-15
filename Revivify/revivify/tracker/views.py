from django.shortcuts import render

# Create your views here.
def phq_form(request):
    print(request.POST)
    if request.method == 'POST':
        ans1 = request.POST['question1']
        print(ans1)
        return render(request, 'tracker/phq9.html') 
      
    else:
        print("thdh")
        return render(request, 'tracker/phq9.html') 
        
  