from django.db import models
import datetime 
# Create your models here.
class Companies(models.Model):
    company_name = models.CharField(max_length=40)
    exchange_company = models.CharField(max_length=40)

class Transacs(models.Model):
    userName = models.CharField(max_length=40)
    amount = models.DecimalField(max_digits=6, decimal_places=2)
    stockValue = models.DecimalField(max_digits=6, decimal_places=2)
    prediction = models.DecimalField(max_digits=6, decimal_places=2)
    stockName = models.CharField(max_length=40)
    date = models.DateField()
    modelType = models.CharField(max_length=40)
    
    
    