from django import forms
from .models import userProfile

class signupForm(forms.ModelForm):
  
        model = userProfile
        fields = ['uname','fname','lname','email','age','gender']
        labels = {
            'uname': 'USERNAME',
            'fname' : 'FIRST NAME',
            'lname' : 'LAST NAME',
            'email': 'EMAIL',
            'age': 'AGE',
            'gender' : 'GENDER',
           
        }