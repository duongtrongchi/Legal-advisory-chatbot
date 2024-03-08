from django.shortcuts import render



def login(request):
    page = "login"
    context = {"page": page}
    return render(request, "users/login_register.html", context=context)

