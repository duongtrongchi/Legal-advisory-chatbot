from django.shortcuts import render



def login(request):
    page = "register"
    context = {"page": page}
    return render(request, "users/login_register.html", context=context)

