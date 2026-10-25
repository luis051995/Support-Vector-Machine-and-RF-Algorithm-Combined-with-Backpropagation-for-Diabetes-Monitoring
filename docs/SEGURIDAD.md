# Protección de la base de datos

## Implementado

- PostgreSQL requerido en modo producción; SQLite queda limitado a pruebas/local.
- ORM Django y parámetros para consultas. Nunca se concatena entrada del paciente en SQL.
- Payload de pacientes, consultas y mediciones cifrado en la aplicación con Fernet (AES-128-CBC + HMAC-SHA256). Incluye nombre, nacimiento, teléfono, antecedentes, notas, lecturas, características y salida del modelo. En la base se almacena un token cifrado, no JSON legible.
- La clave de cifrado no se almacena en la base ni en Git. Es independiente de DJANGO_SECRET_KEY. Se admite una lista de claves para rotación.
- PostgreSQL con TLS `verify-full` y CA del sistema. Se verifica certificado y nombre del servidor, no solo que exista cifrado en tránsito.
- Usuario `polisalud_app` separado del usuario migrador. El servicio HTTP no recibe la contraseña migradora después del arranque. `restrict_db.py` limita permisos y rechaza un usuario runtime elevado.
- La aplicación no tiene permisos de modificar/borrar entradas clínicas, auditoría ni tabla de migraciones. Sin permisos de crear tablas o roles.
- Hash Argon2 para contraseñas, validación de 12 caracteres y contraseña no común. No se cifra reversiblemente la contraseña.
- Protección CSRF en web, cookie HttpOnly, Secure en producción y SameSite Strict. CORS solo para orígenes móviles exactos. El registro público solo crea cuentas paciente y expedientes nuevos; exige CSRF web u origen nativo exacto y limita a cinco intentos por IP por hora.
- Bloqueo de autenticación tras 8 intentos por combinación IP+usuario durante 15 minutos, persistido con django-axes. No es defensa completa frente a ataques distribuidos por muchas identidades.
- Los pacientes solo leen/exportan su expediente. Únicamente administradores pueden consultar el conjunto y crear notas o mediciones. El rol profesional heredado no conserva acceso global.
- Respuestas clínicas `Cache-Control: no-store`, CSP, HSTS y límites de cuerpo de solicitud. DEBUG deshabilitado incluso local.

## Alcance y límites

El cifrado protege una copia de la base sin la clave. No protege frente a un servidor de aplicación comprometido que posea la clave. Permanecen visibles código de expediente, relaciones, autores, tipo de registro, marcas de tiempo y metadatos de auditoría, además de nombres de usuario. No se implementa ocultación de todos los metadatos, MFA, segregación por médico ni RLS de PostgreSQL por paciente; el aislamiento por paciente se aplica en el dominio y la API.

Las exportaciones JSON son legibles por diseño y deben protegerse. La búsqueda por nombre se hace en el navegador sobre la lista de expedientes autorizados, porque la base no indexa nombres en claro. Para volúmenes grandes debe añadirse paginación y un diseño de búsqueda que preserve la privacidad.

No se declara cumplimiento normativo ni certificación clínica. Los planes gratuitos son adecuados para demostración con datos ficticios; usar expedientes reales requiere revisión institucional, contratos y política de privacidad, continuidad de servicio y respaldos.

## Claves y respaldo

Nunca pierdas DATA_ENCRYPTION_KEYS: sin una clave que haya cifrado un registro, ese contenido no se recupera. Restablecer DJANGO_SECRET_KEY invalida sesiones; cambiar la clave de datos sin conservar la anterior hace ilegibles expedientes.

Guarda por separado un respaldo cifrado de PostgreSQL y una copia segura de las claves. Usa `pg_dump` para obtener los valores de payload ya cifrados; `manage.py dumpdata` pasa por el ORM y puede revelar contenido descifrado, por lo que no debe usarse como respaldo clínico.

Para rotar: respalda; configura `DATA_ENCRYPTION_KEYS=NUEVA,ANTERIOR`; ejecuta `USE_MIGRATION_DATABASE=1 python manage.py rotate_clinical_keys --apply` durante mantenimiento; comprueba lectura; conserva la clave antigua para restaurar respaldos viejos. Este comando sí requiere permisos migradores, porque recifra ChartEntry.

Limpia sesiones vencidas con `python manage.py clearsessions`. Los tokens móviles vencidos ya son rechazados aunque sus filas permanezcan; pueden purgarse mediante operación administrativa. Los backups/limpieza programados no se configuran automáticamente en el plan gratuito.

El autorregistro no verifica documentalmente la identidad ni valida correo por enlace. No permite reclamar expedientes existentes mediante nombre, teléfono o ID. Si una persona ya tiene un expediente previo, el administrador debe gestionar la vinculación de su cuenta desde la administración; no se fusionan historiales automáticamente.

## Captura ESP32

Un paciente puede iniciar y finalizar sus propias pruebas de sensor, sin adquirir permisos para editar notas ni expedientes ajenos. El token de dispositivo se muestra una vez y solo permite enviar muestras a una prueba de 10 minutos. No contiene PII ni permite leer expedientes. La señal y configuración se cifran en la base; las muestras y calibraciones se restringen a inserción con el rol PostgreSQL de ejecución. BLE usa emparejamiento cifrado con PIN único, activado físicamente por 60 segundos. No hay atestación de hardware ni garantía de origen biológico de una muestra.
