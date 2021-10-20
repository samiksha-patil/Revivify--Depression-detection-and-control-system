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
        
def dass42(request):
    
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
        ans21 =request.POST['question21']
        ans22= request.POST['question22']
        ans23= request.POST['question23']
        ans24= request.POST['question24']
        ans25= request.POST['question25']
        ans26= request.POST['question26']
        ans27= request.POST['question27']
        ans28= request.POST['question28']
        ans29= request.POST['question29']
        ans30= request.POST['question30']
        ans31= request.POST['question31']
        ans32= request.POST['question32']
        ans33= request.POST['question33']
        ans34= request.POST['question34']
        ans35= request.POST['question35']
        ans36= request.POST['question36']
        ans37= request.POST['question37']
        ans38= request.POST['question38']
        ans39= request.POST['question39']
        ans40= request.POST['question40']
        ans41= request.POST['question41']
        ans42= request.POST['question42']
       

        score=int(ans1)+int(ans2)+int(ans3)+int(ans4)+int(ans5)+int(ans6)+int(ans7)+int(ans8)+int(ans9)+int(ans10)+int(ans11)+int(ans12)+int(ans13)+int(ans14)+int(ans15)+int(ans16)+int(ans17)+int(ans18)+int(ans19)+int(ans20)+int(ans21)+int(ans22)+int(ans23)+int(ans24)+int(ans25)+int(ans26)+int(ans27)+int(ans28)+int(ans29)+int(ans30)+int(ans31)+int(ans32)+int(ans33)+int(ans34)+int(ans35)+int(ans36)+int(ans37)+int(ans38)+int(ans39)+int(ans40)+int(ans41)+int(ans42)        
        depressionscore=int(ans3)+int(ans5)+int(ans10)+int(ans13)+int(ans16)+int(ans17)+int(ans21)+int(ans24)+int(ans26)+int(ans31)+int(ans37)+int(ans38)+int(ans42)
        anxietyscore=int(ans2)+int(ans7)+int(ans9)+int(ans15)+int(ans19)+int(ans20)+int(ans23)+int(ans25)+int(ans28)+int(ans30)+int(ans36)+int(ans40)+int(ans41)
        stressscore=int(ans1)+int(ans6)+int(ans8)+int(ans11)+int(ans12)+int(ans14)+int(ans18)+int(ans22)+int(ans27)+int(ans29)+int(ans32)+int(ans33)+int(ans35)+int(ans39)
        
        print(score)
        #depression
        if depressionscore<=9:
            depressionlevel="normal"
        elif depressionscore <=13:
            depressionlevel="mild "
        elif depressionscore <=20:
            depressionlevel="Moderate "
        elif depressionscore <=27:
            depressionlevel="Severe  "
        else:
            depressionlevel="Very Severe"

        #anxiety
        if anxietyscore<=7:
            anxietylevel="normal"
        elif anxietyscore <=9:
            anxietylevel="mild "
        elif anxietyscore <=14:
            anxietylevel="Moderate "
        elif anxietyscore <=19:
            anxietylevel="Severe  "
        else:
            anxietylevel="Very Severe"

        #stress
        if stressscore<=14:
            stresslevel="normal"
        elif stressscore <=18:
            stresslevel="mild "
        elif stressscore <=25:
            stresslevel="Moderate "
        elif stressscore <=33:
            stresslevel="Severe  "
        else:
            stresslevel="Very Severe"
        

        print(stresslevel)
        print(anxietylevel)
        print( depressionlevel)

        return render(request, 'tracker/dass42_score.html',{
            'score':score,
            'depressionscore':depressionscore,
            'anxietyscore':anxietyscore,
            'stressscore':stressscore,
               'stresslevel':stresslevel,
               'anxietylevel':anxietylevel,
                'depressionlevel': depressionlevel
        }) 


    else:
        return render(request, 'tracker/dass42.html')