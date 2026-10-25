"""Instala esquema y rol limitado desde DATABASE_URL de Neon guardada localmente."""
from pathlib import Path
import os,subprocess,sys
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
subprocess.run([sys.executable,'deploy/prepare_neon.py'],check=True)
load_dotenv(ROOT/'.env',override=False);load_dotenv(ROOT/'.env.local',override=False)
env=os.environ.copy()
if not env.get('MIGRATION_DATABASE_URL') or not env.get('APP_DATABASE_URL'):raise SystemExit('Faltan variables de conexión.')
owner=env.copy();owner['USE_MIGRATION_DATABASE']='1'
steps=[([sys.executable,'manage.py','migrate','--noinput'],owner),([sys.executable,'deploy/restrict_db.py'],env),([sys.executable,'manage.py','verify_neon'],env)]
for cmd,context in steps:
 result=subprocess.run(cmd,env=context)
 if result.returncode:raise SystemExit('Configuración de Neon interrumpida. Corrige el error anterior; tus claves y base local se conservan.')
print('NEON_SETUP_OK: esquema migrado, cuenta limitada y TLS verificados. No se importaron pacientes desde SQLite.')
