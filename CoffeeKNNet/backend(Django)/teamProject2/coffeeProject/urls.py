from django.urls import path

from coffeeProject import views

urlpatterns = [
    path('coffeePredict', views.coffeePredict),

]