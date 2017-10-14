from django.conf.urls import url
from . import views


app_name = 'housing'

urlpatterns = [
    url(r'^$', views.IndexView, name="index"),
    url(r'predict/$', views.Predict, name="predict"),
    url(r'predictreg/$', views.PredictReg, name="predictreg"),
    # url(r'plots/$', views.plots, name="plots"),

]