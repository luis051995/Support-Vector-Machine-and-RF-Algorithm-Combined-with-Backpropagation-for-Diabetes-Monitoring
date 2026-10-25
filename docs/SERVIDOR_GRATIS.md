# Django en Render Free + PostgreSQL en Neon Free

Guía verificada con documentación oficial el 24 de octubre de 2026. Es una opción de demostración, sujeta a cambios de cuotas y condiciones de los proveedores. No se crearon cuentas ni infraestructura externa del usuario durante esta entrega.

Render Free hospeda Django y ofrece URL HTTPS. Se suspende tras 15 minutos sin tráfico y puede tardar alrededor de un minuto en despertar. El almacenamiento local es efímero: no guardar SQLite ni subir adjuntos al disco. Render indica no usar instancias Free para aplicaciones de producción.

Neon ofrece PostgreSQL con plan Free y cuotas de almacenamiento/cómputo. La configuración usa una base NUEVA y dedicada a este proyecto, que conserva los registros aunque Render reinicie. No se utiliza Render Postgres Free, cuya base expira a los 30 días. Comprueba las cuotas vigentes al registrarte y conserva respaldos externos.

## 1. Preparar repositorio

Descomprime el paquete y sube la carpeta `polisalud_django` a un repositorio privado propio de GitHub, con su contenido como raíz. El archivo render.yaml debe estar en la raíz del repositorio. No subas .env, claves, datos de pacientes, archivos .sqlite3 ni respaldos. El .gitignore incluido los excluye, pero revisa antes de publicar. Los archivos `data/diabetes_original.csv` son los datos de investigación aportados, separados de los expedientes operativos; valida su autorización de uso antes de subirlos a cualquier proveedor.

## 2. Crear PostgreSQL

En https://neon.com crea un proyecto Free y una base dedicada, por ejemplo `polisalud`. Obtén la conexión directa del propietario (sin pooler): se usa como MIGRATION_DATABASE_URL. No la pegues en un chat, código ni repositorio.

Genera una contraseña independiente para el usuario `polisalud_app`. Construye DATABASE_URL con el mismo host y nombre de base, usuario `polisalud_app` y esa contraseña. Usa una contraseña alfanumérica aleatoria larga para evitar errores de escape de URL; si contiene caracteres especiales codifícalos correctamente. El rol lo crea el script de arranque utilizando la cuenta propietaria; esta debe tener permiso para crear roles. Si el proveedor no lo permite, crea el rol desde su panel y vuelve a desplegar.

Ambas URLs deben apuntar al mismo endpoint directo y base. Se evita conexión pooler en esta versión. El arranque valida el usuario limitado, migra con la cuenta propietaria y aplica GRANT/REVOKE. Esa rutina se diseñó para una base vacía y dedicada, no una base compartida con otras aplicaciones.

## 3. Configurar Render

En https://render.com conecta tu repositorio privado y crea un Blueprint con render.yaml. Selecciona el plan Free y revisa cualquier solicitud de pago antes de aceptarla. El archivo no crea productos de pago ni una base Render.

Introduce estos secretos en el panel de Render:

| Variable | Valor |
|---|---|
| DJANGO_SECRET_KEY | Clave aleatoria de al menos 50 caracteres |
| DATA_ENCRYPTION_KEYS | Clave Fernet generada localmente, guardada en tu gestor de secretos |
| DATABASE_URL | Conexión de polisalud_app |
| MIGRATION_DATABASE_URL | Conexión propietaria, distinta de runtime |
| BOOTSTRAP_ADMIN_USERNAME | Usuario administrador inicial |
| BOOTSTRAP_ADMIN_PASSWORD | Contraseña fuerte inicial, al menos 12 caracteres |

Para generar claves en tu equipo, ejecuta `python setup_local.py` una sola vez en una carpeta local de trabajo o genera claves nuevas en tu gestor de secretos. Para hosting usa secretos diferentes a los de desarrollo. No compartas la pantalla ni la salida con secretos.

APP_ENV=production y TRUST_RENDER_PROXY=1 ya están configurados. El hostname y el origen CSRF de Render se toman de RENDER_EXTERNAL_HOSTNAME. Si añades dominio propio define DJANGO_ALLOWED_HOSTS y CSRF_TRUSTED_ORIGINS con los valores exactos.

El build instala dependencias, recoge estáticos y ejecuta check --deploy. El start aplica migraciones, configura permisos mínimos, crea el primer administrador si falta y ejecuta Gunicorn con un worker y dos hilos. El modelo se carga bajo demanda para reducir memoria inicial. El consumo real con IA debe probarse contra la RAM del plan Free; no se ha medido en Render y no se garantiza que la cuota sea suficiente para toda carga.

## 4. Verificar

Abre tu URL de Render, inicia sesión y crea únicamente un paciente ficticio. Registra consulta/medición; vuelve a desplegar y confirma que permanecen. Prueba una cuenta paciente y verifica que no puede consultar otro expediente. Abre /api/health para comprobar disponibilidad.

Después del primer inicio quita BOOTSTRAP_ADMIN_PASSWORD del panel de Render y cambia tu contraseña en Mi cuenta. Conserva MIGRATION_DATABASE_URL como secreto de arranque para futuras migraciones; el proceso HTTP la elimina de su entorno antes de ejecutarse.

## 5. Móvil

Pon la URL HTTPS de Render en POLISALUD_API_URL y ejecuta la preparación de assets y sincronización Capacitor; ver mobile/README.md. Los móviles usan el mismo Django y PostgreSQL. Las aplicaciones firmadas de iOS requieren cuenta/herramientas Apple; el hosting gratuito no elimina esos requisitos.

## Estado de entrega

El proyecto y archivos de despliegue están preparados. El hosting Django no está publicado en una cuenta Render/Neon porque no hay acceso autorizado a esas cuentas ni a sus conexiones de base de datos. La demostración web anterior es independiente y no se convirtió silenciosamente en este backend.

Fuentes:
- https://render.com/docs/free
- https://render.com/docs/deploy-django
- https://neon.com/docs/introduction/plans
- https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/
