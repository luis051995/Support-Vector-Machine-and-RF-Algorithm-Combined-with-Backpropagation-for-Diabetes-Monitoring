from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User
@admin.register(User)
class AccountsAdmin(UserAdmin):
    fieldsets=UserAdmin.fieldsets+(("PoliSalud",{'fields':('role','patient')}),)
    add_fieldsets=UserAdmin.add_fieldsets+(("PoliSalud",{'fields':('role','patient')}),)
    list_display=('username','role','is_active','is_staff')
