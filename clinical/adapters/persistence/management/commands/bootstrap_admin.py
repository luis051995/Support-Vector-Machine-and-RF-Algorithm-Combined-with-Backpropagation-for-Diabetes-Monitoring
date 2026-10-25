import os
from django.core.management.base import BaseCommand,CommandError
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from identity.models import User
class Command(BaseCommand):
    help='Crea admin inicial desde variables solo si no hay administradores; nunca cambia contraseñas existentes.'
    def handle(self,*a,**k):
        username=os.getenv('BOOTSTRAP_ADMIN_USERNAME');password=os.getenv('BOOTSTRAP_ADMIN_PASSWORD')
        if not username or not password:return
        if User.objects.filter(is_superuser=True).exists():self.stdout.write('Administrador ya inicializado; no se modifica.');return
        u=User(username=username,role='admin')
        try:validate_password(password,u)
        except ValidationError:raise CommandError('BOOTSTRAP_ADMIN_PASSWORD no cumple la política de contraseña.')
        User.objects.create_superuser(username=username,password=password,role='admin');self.stdout.write('Administrador inicial creado. Retira BOOTSTRAP_ADMIN_PASSWORD del hosting.')
