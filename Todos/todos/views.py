from django.shortcuts import HttpResponse, redirect, render
from .models import Todo

# Create your views here.

def home(request):
    todos = Todo.objects.all()
    name = "Bidur Siwakoti"
    return render(request,'index.html', {'todos': todos, 'abc': name})

def add_todo(request):
    if request.method == "POST":
        title = request.POST['title']
        description = request.POST['description']
        new_todo = Todo(title=title, description=description)
        new_todo.save()
        return redirect('/home')
    
def update(request, todo_id):
    todo = Todo.objects.get(pk = todo_id)
    return render(request, 'update.html', {'todo': todo})


def supdate(request, todo_id):
    todo = Todo.objects.get(pk = todo_id)
    todo.title = request.POST['title']
    todo.description = request.POST['description']
    todo.save()
    return redirect('/home')

def delete(request, todo_id):
    todo = Todo.objects.get(pk = todo_id)
    todo.delete()
    return redirect('/home')



# def service(request):
#     todos = Todo.objects.all()
#     return render(request,'service.html')


# def about(request):
#     todos = Todo.objects.all()
#     return render(request,'about.html')