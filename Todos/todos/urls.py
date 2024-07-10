from django.urls import path
from .views import add_todo, home, update,supdate, delete

urlpatterns = [
    path('home/', home, name = 'home'),
    path('add_todo/', add_todo, name='add_todo'),
    path('Update/<int:todo_id>', update, name ='show_update' ),
    path('supdate/<int:todo_id>', supdate, name = 'supdate' ),
    path('delete/<int:todo_id>', delete, name ='delete')
    
]