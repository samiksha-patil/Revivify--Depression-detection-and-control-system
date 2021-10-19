from django.shortcuts import render

# Create your views here.
def phq_form(request):
    print(request.POST)
    if request.method == 'POST':
        ans1 =request.POST['question1']
        ans2= request.POST['question2']
        ans3= request.POST['question3']
        ans4= request.POST['question4']
        ans5= request.POST['question5']
        ans6= request.POST['question6']
        ans7= request.POST['question7']
        ans8= request.POST['question8']
        ans9= request.POST['question9']
        score=int(ans1)+int(ans2)+int(ans3)+int(ans4)+int(ans5)+int(ans6)+int(ans7)+int(ans8)+int(ans9)
        print(score)
        if score<=4:
            level="mild depression"
        elif score <=9:
            level="mild depression"
        elif score <=14:
            level="Moderate depression"
        elif score <=19:
            level="Moderately severe depression "
        elif score <=27:
            level="Severe depression"
        print(level)
        return render(request, 'tracker/phq9.html',{
            'level':level,          
        }) 

    else:
        print("thdh")
        return render(request, 'tracker/phq9.html') 

def dass21_form(request):
    print(request.POST)
    if request.method == 'POST':
        ans1 =request.POST['question1']
        ans2= request.POST['question2']
        ans3= request.POST['question3']
        ans4= request.POST['question4']
        ans5= request.POST['question5']
        ans6= request.POST['question6']
        ans7= request.POST['question7']
        ans8= request.POST['question8']
        ans9= request.POST['question9']
        ans10 =request.POST['question10']
        ans11 =request.POST['question11']
        ans12= request.POST['question12']
        ans13= request.POST['question13']
        ans14= request.POST['question14']
        ans15= request.POST['question15']
        ans16= request.POST['question16']
        ans17= request.POST['question17']
        ans18= request.POST['question18']
        ans19= request.POST['question19']
        ans20= request.POST['question20']
        ans21= request.POST['question21']

        score=int(ans1)+int(ans2)+int(ans3)+int(ans4)+int(ans5)+int(ans6)+int(ans7)+int(ans8)+int(ans9)
        print(score)
        if score<=4:
            level="mild depression"
        elif score <=9:
            level="mild depression"
        elif score <=14:
            level="Moderate depression"
        elif score <=19:
            level="Moderately severe depression "
        elif score <=27:
            level="Severe depression"
        print(level)
        return render(request, 'tracker/dass21_form.html',{
            'level':level,          
        }) 

    else:
        print("thdh")
        return render(request, 'tracker/dass21_form.html') 
        
  