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
        return render(request, 'tracker/phq9score.html',{
            'level':level, 
            'score':score         
        }) 

    else:
        print("thdh")
        return render(request, 'tracker/phq9.html') 
        
  