from django import forms
from .models import userProfile

class signupForm(forms.ModelForm):
    class Meta:
        model = userProfile
        fields = ['username','fname','lname','email','age','gender']
        labels = {
            'username' : 'USERNAME',
            'fname' : 'FIRST NAME',
            'lname' : 'LAST NAME',
            'age': 'AGE',
            'gender' : 'GENDER',
           
        }