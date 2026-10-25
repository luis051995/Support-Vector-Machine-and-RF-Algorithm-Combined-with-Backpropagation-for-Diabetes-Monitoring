import os,subprocess,sys
# Solo el subproceso de migraciones recibe la conexión propietaria.
env=os.environ.copy()
if not env.get('MIGRATION_DATABASE_URL'):raise SystemExit('MIGRATION_DATABASE_URL requerida; separar de DATABASE_URL de la aplicación.')
env['USE_MIGRATION_DATABASE']='1'
subprocess.run([sys.executable,'manage.py','migrate','--noinput'],check=True,env=env)
subprocess.run([sys.executable,'deploy/restrict_db.py'],check=True,env=os.environ.copy())
subprocess.run([sys.executable,'manage.py','bootstrap_admin'],check=True)
# Los workers HTTP no heredan las credenciales de migración ni contraseña inicial.
for name in ['MIGRATION_DATABASE_URL','BOOTSTRAP_ADMIN_PASSWORD']:os.environ.pop(name,None)
os.execv(sys.executable,[sys.executable,'-m','gunicorn','config.wsgi:application','--bind','0.0.0.0:'+os.environ.get('PORT','8000'),'--workers','1','--threads','2','--timeout','120','--access-logfile','-'])
