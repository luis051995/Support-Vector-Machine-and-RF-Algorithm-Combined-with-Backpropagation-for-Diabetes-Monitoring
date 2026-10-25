"""Normaliza variables CLI para Django sin imprimir ni subir credenciales."""
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,quote
import secrets,sys
from dotenv import dotenv_values,set_key
ROOT=Path(__file__).resolve().parents[1]
env=ROOT/'.env';source=env if env.exists() else ROOT/'.env.local'
values=dotenv_values(source)
uri=values.get('DATABASE_URL')
if not uri:raise SystemExit('Falta DATABASE_URL: ejecuta neon env pull o configúrala de forma privada.')
p=urlsplit(uri)
if not p.hostname or not p.hostname.endswith('.neon.tech'):raise SystemExit('DATABASE_URL no corresponde a Neon.')
direct=values.get('DATABASE_URL_UNPOOLED') or urlunsplit(p._replace(netloc=p.netloc.replace(p.hostname,p.hostname.replace('-pooler',''))))
set_key(str(source),'DATABASE_URL_UNPOOLED',direct)
set_key(str(source),'MIGRATION_DATABASE_URL',direct)
if not values.get('APP_DATABASE_URL'):
 password=secrets.token_urlsafe(48)
 host=p.hostname if '-pooler' in p.hostname else p.hostname.replace('.c-','-pooler.c-',1)
 # La entrada del CLI normalmente ya es pooled; sin el patrón .c-, usar endpoint directo.
 port=':'+str(p.port) if p.port else ''
 app=urlunsplit(p._replace(netloc='polisalud_app:'+quote(password,safe='')+'@'+host+port))
 set_key(str(source),'APP_DATABASE_URL',app)
source.chmod(0o600)
print('NEON_ENV_PREPARED: conexión directa para migraciones y credencial separada polisalud_app. Aún requiere crear el rol y migrar en Neon.')
