from django.db import models
from django.contrib.auth.models import User
from django.db.models import DO_NOTHING
from django.conf import settings

GENDER = (

    ('F','female'),
    ('M','male'),  
    ('other','other')
)


class userProfile(models.Model):
    user = models.CharField(max_length=50)
    fname = models.CharField(max_length=50)
    email = models.CharField(max_length=100)
    lname = models.CharField(max_length=50)
    age =   models.IntegerField(blank=True, null=True)
    gender = models.CharField('Gender',choices=GENDER,max_length=12)
    
