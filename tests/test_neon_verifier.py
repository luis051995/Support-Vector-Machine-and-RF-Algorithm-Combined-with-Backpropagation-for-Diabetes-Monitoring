from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock,patch
from django.core.management.base import CommandError
from clinical.adapters.persistence.management.commands.verify_neon import Command
class NeonVerifierTests(TestCase):
    def fake(self,tls):
        cursor=Mock();cursor.fetchone.side_effect=[('neondb','polisalud_app'),(0,),(False,)]
        db=SimpleNamespace(vendor='postgresql',connection=SimpleNamespace(pgconn=SimpleNamespace(ssl_in_use=tls)),cursor=lambda:nullcontext(cursor))
        return db,cursor
    def test_pooler_uses_client_tls_state(self):
        db,cursor=self.fake(True);out=StringIO()
        with patch('clinical.adapters.persistence.management.commands.verify_neon.connection',db):Command(stdout=out).handle()
        self.assertIn('NEON_DATABASE_OK',out.getvalue())
        self.assertFalse(any('pg_stat_ssl' in str(c) for c in cursor.execute.call_args_list))
    def test_unencrypted_client_is_rejected(self):
        db,_=self.fake(False)
        with patch('clinical.adapters.persistence.management.commands.verify_neon.connection',db):
            with self.assertRaises(CommandError):Command(stdout=StringIO()).handle()
