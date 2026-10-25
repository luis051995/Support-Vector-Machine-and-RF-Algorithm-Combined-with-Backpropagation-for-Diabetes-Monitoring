"""Configura permisos mínimos en una base PostgreSQL NUEVA y dedicada.
Las credenciales se leen únicamente de variables, nunca se imprimen.
"""
import os
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
owner=os.environ.get('MIGRATION_DATABASE_URL');runtime=os.environ.get('DATABASE_URL')
if not owner or not runtime:raise SystemExit('Configura ambas URLs de PostgreSQL.')
a=conninfo_to_dict(owner);b=conninfo_to_dict(runtime)
if b.get('user')!='polisalud_app' or a.get('user')==b.get('user'):raise SystemExit('DATABASE_URL debe usar polisalud_app y la cuenta migradora debe ser distinta.')
if a.get('host')!=b.get('host') or a.get('dbname')!=b.get('dbname'):raise SystemExit('Ambas conexiones deben apuntar al mismo endpoint y base dedicada.')
if not b.get('password'):raise SystemExit('La cuenta runtime requiere contraseña propia.')
with psycopg.connect(owner,sslmode='verify-full',sslrootcert='system',connect_timeout=10) as c:
    role=c.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolbypassrls FROM pg_roles WHERE rolname=%s',('polisalud_app',)).fetchone()
    membership=c.execute('SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON r.oid=m.member WHERE r.rolname=%s LIMIT 1',('polisalud_app',)).fetchone()
    if membership:raise SystemExit('polisalud_app no debe ser miembro de otros roles; crea un usuario runtime dedicado sin membresías.')
    if role and any(role):raise SystemExit('La cuenta runtime tiene privilegios elevados; configura otra cuenta sin ellos.')
    if role:c.execute(sql.SQL('ALTER ROLE polisalud_app PASSWORD {}').format(sql.Literal(b['password'])))
    else:c.execute(sql.SQL('CREATE ROLE polisalud_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD {}').format(sql.Literal(b['password'])))
    c.execute('REVOKE CREATE ON SCHEMA public FROM PUBLIC')
    c.execute('GRANT USAGE ON SCHEMA public TO polisalud_app')
    c.execute('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO polisalud_app')
    c.execute('GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public TO polisalud_app')
    c.execute('REVOKE UPDATE,DELETE,TRUNCATE ON records_auditevent,records_chartentry,records_sensorsample,records_sensorcalibration FROM polisalud_app')
    c.execute('REVOKE ALL ON django_migrations FROM polisalud_app')
print('Permisos runtime aplicados; bitácora y entradas clínicas sin UPDATE/DELETE.')
