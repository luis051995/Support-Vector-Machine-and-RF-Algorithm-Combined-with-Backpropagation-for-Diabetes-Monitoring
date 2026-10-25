from django.apps import AppConfig
class RecordsConfig(AppConfig):
    name='clinical.adapters.persistence'
    label='records'
    default_auto_field='django.db.models.BigAutoField'
    def ready(self):
        from . import checks
