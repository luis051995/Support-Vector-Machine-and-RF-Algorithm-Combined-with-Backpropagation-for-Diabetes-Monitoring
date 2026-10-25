from django.core.management.base import BaseCommand,CommandError
from django.db import connection
class Command(BaseCommand):
    help='Prueba real y de solo lectura: PostgreSQL, TLS, rol limitado y tablas.'
    def handle(self,*a,**k):
        if connection.vendor!='postgresql':raise CommandError('No se está usando PostgreSQL. Esta prueba no acepta SQLite como evidencia de Neon.')
        try:
            with connection.cursor() as c:
                c.execute('SELECT current_database(),current_user');database,user=c.fetchone()
                # Verifica TLS del cliente; pg_stat_ssl solo ve el tramo del backend.
                if not connection.connection.pgconn.ssl_in_use:raise CommandError('La conexión del cliente no tiene TLS.')
                if user!='polisalud_app':raise CommandError('La aplicación no usa el rol limitado polisalud_app.')
                c.execute('SELECT COUNT(*) FROM records_patient');count=c.fetchone()[0]
                c.execute("SELECT has_table_privilege(current_user,'records_sensorsample','UPDATE') OR has_table_privilege(current_user,'records_auditevent','DELETE')")
                if c.fetchone()[0]:raise CommandError('Permisos de escritura histórica demasiado amplios.')
        except CommandError:raise
        except Exception as e:raise CommandError('No se pudo verificar Neon ('+type(e).__name__+'). Revisa red, TLS, credenciales y migraciones; no compartas la URL completa.')
        self.stdout.write(self.style.SUCCESS(f'NEON_DATABASE_OK database={database} role={user} TLS=true patients={count} historical_mutation=false'))
