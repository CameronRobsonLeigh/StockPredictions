from django.urls import path
from . import views
from django.urls import re_path as url

urlpatterns = [
    path('', views.Index, name='Index'),
    path('IndexCustom/', views.IndexCustom, name='IndexCustom'),
    path('GenerateNeuralNetwork/', views.GenerateNeuralNetwork, name='GenerateNeuralNetwork'),
    path("Register/", views.register_request, name="Register"),
    path("Login/", views.login_request, name="Login"),
    path("logout/", views.logout_request, name= "logout"),
    path('Transactions/', views.Transactions, name= "Transactions"),
    path("PlaceTransaction/", views.PlaceTransaction, name= "PlaceTransaction")
]