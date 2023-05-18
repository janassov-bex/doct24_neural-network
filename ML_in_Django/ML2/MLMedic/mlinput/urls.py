from django.urls import path
from mlinput.views import *

urlpatterns = [
    path('lung_cancer/', predict_lungcancer, name='lung_caner'),
    path('lung_cancer/lung_cancer_answer/', answer_lungcancer, name='lung_cancer_answer'),
    path('heartattack/', predict_heartattack, name='heartattck'),
    path('heartattack/heartattack_answer/', answer_heartattack, name='heartattack_answer'),
    path('test/', test, name='test')
]
