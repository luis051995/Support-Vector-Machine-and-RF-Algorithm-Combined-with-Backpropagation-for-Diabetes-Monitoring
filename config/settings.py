import os
from pathlib import Path
from datetime import timedelta
import dj_database_url
from django.core.exceptions import ImproperlyConfigured
BASE_DIR=Path(__file__).resolve().parents[1]
# Se carga .env solo como configuración explícita del operador, nunca se versiona.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR/'.env',override=False)
except ImportError:pass
MODE=os.getenv('APP_ENV','development')
PRODUCTION=MODE=='production'
DEBUG=False
SECRET_KEY=os.getenv('DJANGO_SECRET_KEY','')
DATA_ENCRYPTION_KEYS=[k.strip() for k in os.getenv('DATA_ENCRYPTION_KEYS','').split(',') if k.strip()]
if not SECRET_KEY or len(SECRET_KEY)<50:raise ImproperlyConfigured('Configura DJANGO_SECRET_KEY (al menos 50 caracteres) en .env o variables de entorno.')
if not DATA_ENCRYPTION_KEYS:raise ImproperlyConfigured('Configura DATA_ENCRYPTION_KEYS. No regeneres la clave si ya existen datos.')
ALLOWED_HOSTS=[x.strip() for x in os.getenv('DJANGO_ALLOWED_HOSTS','localhost,127.0.0.1,testserver' if not PRODUCTION else '').split(',') if x.strip()]
render_host=os.getenv('RENDER_EXTERNAL_HOSTNAME')
if render_host and render_host not in ALLOWED_HOSTS:ALLOWED_HOSTS.append(render_host)
if PRODUCTION and (not ALLOWED_HOSTS or '*' in ALLOWED_HOSTS):raise ImproperlyConfigured('Define hosts de producción exactos; no se admite *.')
INSTALLED_APPS=['django.contrib.admin','django.contrib.auth','django.contrib.contenttypes','django.contrib.sessions','django.contrib.messages','django.contrib.staticfiles','rest_framework','corsheaders','axes','identity','clinical.adapters.persistence.apps.RecordsConfig']
MIDDLEWARE=['django.middleware.security.SecurityMiddleware','whitenoise.middleware.WhiteNoiseMiddleware','corsheaders.middleware.CorsMiddleware','django.contrib.sessions.middleware.SessionMiddleware','django.middleware.common.CommonMiddleware','django.middleware.csrf.CsrfViewMiddleware','django.contrib.auth.middleware.AuthenticationMiddleware','django.contrib.messages.middleware.MessageMiddleware','django.middleware.clickjacking.XFrameOptionsMiddleware','axes.middleware.AxesMiddleware','config.middleware.SecurityHeadersMiddleware']
ROOT_URLCONF='config.urls'
TEMPLATES=[{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':[BASE_DIR/'templates'],'APP_DIRS':True,'OPTIONS':{'context_processors':['django.template.context_processors.request','django.contrib.auth.context_processors.auth','django.contrib.messages.context_processors.messages']}}]
WSGI_APPLICATION='config.wsgi.application'
AUTH_USER_MODEL='identity.User'
AUTHENTICATION_BACKENDS=['axes.backends.AxesStandaloneBackend','django.contrib.auth.backends.ModelBackend']
AXES_FAILURE_LIMIT=8
AXES_COOLOFF_TIME=timedelta(minutes=15)
AXES_LOCKOUT_PARAMETERS=[['username','ip_address']]
AXES_RESET_ON_SUCCESS=True
AXES_HTTP_RESPONSE_CODE=429
# IP sin confiar en cabeceras enviadas por el cliente. Evita suplantar X-Forwarded-For.
AXES_IPWARE_PROXY_COUNT=0
AXES_CLIENT_IP_CALLABLE='config.security.client_ip'
AXES_SENSITIVE_PARAMETERS=['username','password']
DATABASE_URL=os.getenv('DATABASE_URL')
if os.getenv('USE_MIGRATION_DATABASE')=='1':DATABASE_URL=os.getenv('MIGRATION_DATABASE_URL') or DATABASE_URL
if PRODUCTION and not DATABASE_URL:raise ImproperlyConfigured('Producción requiere PostgreSQL mediante DATABASE_URL; SQLite solo para desarrollo.')
if DATABASE_URL:
    DATABASES={'default':dj_database_url.parse(DATABASE_URL,conn_max_age=0,conn_health_checks=True)}
    if PRODUCTION:
        if DATABASES['default']['ENGINE']!='django.db.backends.postgresql':raise ImproperlyConfigured('Solo PostgreSQL en producción.')
        # La CA del sistema debe validar el certificado y el nombre del servidor.
        DATABASES['default'].setdefault('OPTIONS',{}).update({'sslmode':'verify-full','sslrootcert':'system','connect_timeout':10})
        DATABASES['default']['DISABLE_SERVER_SIDE_CURSORS']=True
else:DATABASES={'default':{'ENGINE':'django.db.backends.sqlite3','NAME':BASE_DIR/'instance.sqlite3'}}
AUTH_PASSWORD_VALIDATORS=[{'NAME':'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},{'NAME':'django.contrib.auth.password_validation.MinimumLengthValidator','OPTIONS':{'min_length':12}},{'NAME':'django.contrib.auth.password_validation.CommonPasswordValidator'},{'NAME':'django.contrib.auth.password_validation.NumericPasswordValidator'}]
PASSWORD_HASHERS=['django.contrib.auth.hashers.Argon2PasswordHasher','django.contrib.auth.hashers.PBKDF2PasswordHasher']
REST_FRAMEWORK={'DEFAULT_AUTHENTICATION_CLASSES':['identity.authentication.ExpiringBearerAuthentication','rest_framework.authentication.SessionAuthentication'],'DEFAULT_PERMISSION_CLASSES':['rest_framework.permissions.IsAuthenticated'],'DEFAULT_PARSER_CLASSES':['clinical.adapters.http.parsers.ObjectJSONParser'],'DEFAULT_RENDERER_CLASSES':['rest_framework.renderers.JSONRenderer'],'EXCEPTION_HANDLER':'clinical.adapters.http.errors.exception_handler'}
SESSION_COOKIE_AGE=8*3600
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SECURE=PRODUCTION
CSRF_COOKIE_SECURE=PRODUCTION
SESSION_COOKIE_SAMESITE='Strict'
CSRF_COOKIE_SAMESITE='Strict'
CSRF_TRUSTED_ORIGINS=[x for x in os.getenv('CSRF_TRUSTED_ORIGINS','').split(',') if x]
if render_host:CSRF_TRUSTED_ORIGINS.append('https://'+render_host)
CORS_ALLOWED_ORIGINS=[x for x in os.getenv('MOBILE_ALLOWED_ORIGINS','capacitor://localhost,https://localhost,http://localhost').split(',') if x]
CORS_ALLOW_CREDENTIALS=False
SECURE_SSL_REDIRECT=PRODUCTION
SECURE_HSTS_SECONDS=31536000 if PRODUCTION else 0
SECURE_HSTS_INCLUDE_SUBDOMAINS=PRODUCTION
SECURE_HSTS_PRELOAD=PRODUCTION
# Habilitar solo detrás del proxy administrado de Render, que fija este header.
if os.getenv('TRUST_RENDER_PROXY')=='1':SECURE_PROXY_SSL_HEADER=('HTTP_X_FORWARDED_PROTO','https')
SECURE_CONTENT_TYPE_NOSNIFF=True
SECURE_REFERRER_POLICY='same-origin'
X_FRAME_OPTIONS='DENY'
DATA_UPLOAD_MAX_MEMORY_SIZE=128*1024
STATIC_URL='/static/'
STATIC_ROOT=BASE_DIR/'staticfiles'
STATICFILES_DIRS=[BASE_DIR/'web']
STORAGES={'default':{'BACKEND':'django.core.files.storage.FileSystemStorage'},'staticfiles':{'BACKEND':'whitenoise.storage.CompressedManifestStaticFilesStorage'}}
LANGUAGE_CODE='es-mx'
TIME_ZONE='America/Mexico_City'
USE_I18N=True
USE_TZ=True
DEFAULT_AUTO_FIELD='django.db.models.BigAutoField'
DEFAULT_EXCEPTION_REPORTER_FILTER='config.middleware.RedactingExceptionFilter'
LOGGING={'version':1,'disable_existing_loggers':False,'handlers':{'console':{'class':'logging.StreamHandler'}},'root':{'handlers':['console'],'level':'WARNING'},'loggers':{'django.db.backends':{'handlers':['console'],'level':'ERROR','propagate':False}}}

CSRF_FAILURE_VIEW='config.security.csrf_failure'
# Sonda sin datos de paciente; admite health checks internos HTTP de Render.
SECURE_REDIRECT_EXEMPT=[r'^api/health$']
