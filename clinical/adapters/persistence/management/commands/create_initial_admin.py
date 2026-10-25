from django.core.management.base import BaseCommand
from identity.models import User
from django.core.management import call_command
class Command(BaseCommand):
    help='Solicita crear un superusuario solo si la instalación no tiene uno.'
    def handle(self,*a,**k):
        if User.objects.filter(is_superuser=True).exists():self.stdout.write('Administrador existente; se conserva.')
        else:call_command('createsuperuser',interactive=True)
