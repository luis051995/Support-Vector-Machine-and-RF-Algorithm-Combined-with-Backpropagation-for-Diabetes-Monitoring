from django.core.management.base import BaseCommand,CommandError
from django.conf import settings
from django.db import transaction
from clinical.adapters.persistence.models import Patient,ChartEntry
class Command(BaseCommand):
    help='Recifra payloads con la primera clave; requiere clave anterior en DATA_ENCRYPTION_KEYS. Ejecutar con cuenta migradora y respaldo previo.'
    def add_arguments(self,parser):parser.add_argument('--apply',action='store_true')
    def handle(self,*args,**options):
        if not options['apply']:raise CommandError('Respaldar base y claves. Añade nueva clave primero y conserva anteriores; usa --apply para recifrar.')
        count=0
        with transaction.atomic():
            for model in (Patient,ChartEntry):
                for row in model.objects.select_for_update().iterator(chunk_size=100):row.save(update_fields=['payload']);count+=1
        self.stdout.write(self.style.SUCCESS(f'{count} payloads recifrados. Conserva claves antiguas para respaldos anteriores.'))
