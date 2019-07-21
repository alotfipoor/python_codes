from . import views
from django.urls import path, re_path

urlpatterns = [
    path('', views.index, name='index'),
    re_path(r'^details/(?P<id>\d+)/$', views.details, name='details'),
]
