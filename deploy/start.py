import os,subprocess,sys
# Solo el subproceso de migraciones recibe la conexión propietaria.
env=os.environ.copy()
if not env.get('MIGRATION_DATABASE_URL'):env['MIGRATION_DATABASE_URL']=env.get('DATABASE_URL_UNPOOLED','')
if not env.get('MIGRATION_DATABASE_URL'):raise SystemExit('Falta conexión directa de migración.')
if not env.get('APP_DATABASE_URL'):raise SystemExit('Configura APP_DATABASE_URL con el usuario limitado polisalud_app.')
os.environ['MIGRATION_DATABASE_URL']=env['MIGRATION_DATABASE_URL']
env['USE_MIGRATION_DATABASE']='1'
subprocess.run([sys.executable,'manage.py','migrate','--noinput'],check=True,env=env)
subprocess.run([sys.executable,'deploy/restrict_db.py'],check=True,env=os.environ.copy())
subprocess.run([sys.executable,'manage.py','bootstrap_admin'],check=True)
# Los workers HTTP no heredan las credenciales de migración ni contraseña inicial.
# El proceso web conserva únicamente la conexión del usuario limitado.
os.environ['DATABASE_URL']=os.environ['APP_DATABASE_URL']
for name in ['MIGRATION_DATABASE_URL','DATABASE_URL_UNPOOLED','BOOTSTRAP_ADMIN_PASSWORD']:os.environ.pop(name,None)
os.execv(sys.executable,[sys.executable,'-m','gunicorn','config.wsgi:application','--bind','0.0.0.0:'+os.environ.get('PORT','8000'),'--workers','1','--threads','2','--timeout','120','--access-logfile','-'])
