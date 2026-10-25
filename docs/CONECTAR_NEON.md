# Conectar Django a Neon con una URL existente

No se coloca la contraseña en settings.py ni en Git. Conserva .env, DJANGO_SECRET_KEY y DATA_ENCRYPTION_KEYS de tu instalación actual.

## En Windows

1. Actualiza los archivos del proyecto con esta versión, conservando .env, .venv e instance.sqlite3.
2. Abre PowerShell en la carpeta que contiene manage.py y ejecuta `notepad .env`.
3. Añade o reemplaza `DATABASE_URL` con la conexión PostgreSQL real, entre comillas. No cambies las otras claves. No compartas esa línea.
4. Ejecuta `.\.venv\Scripts\python.exe deploy\setup_neon.py`.
5. Continúa solo cuando aparezcan NEON_DATABASE_OK y NEON_SETUP_OK.
6. Crea el administrador mediante `.\.venv\Scripts\python.exe manage.py create_initial_admin`, recoge estáticos y arranca normalmente.

setup_neon.py prepara las variables, aplica migraciones con conexión directa, crea/configura polisalud_app con contraseña independiente, limita permisos y comprueba TLS/tablas usando la cuenta de aplicación. Requiere que la cuenta propietaria pueda migrar y crear roles. Se diseñó para una base dedicada a PoliSalud. No importa automáticamente los datos de SQLite ni cambia las claves de cifrado.

Variables:

| Variable | Finalidad |
|---|---|
| DATABASE_URL | Conexión administrada por Neon/CLI, generalmente pooled |
| DATABASE_URL_UNPOOLED | Variante directa para operaciones de esquema |
| MIGRATION_DATABASE_URL | Conexión propietaria directa para migrar |
| APP_DATABASE_URL | Conexión de polisalud_app; Django la usa preferentemente |

El contenido de .env prevalece sobre .env.local y las variables del proceso prevalecen sobre ambos. El arranque productivo exige APP_DATABASE_URL y elimina las conexiones propietarias del entorno del proceso HTTP.

## CLI y MCP

Neon CLI 8.11.3 se utilizó mediante npx porque la instalación global falló con EACCES en este entorno. El init instaló el plugin Neon para Codex, con MCP HTTP https://mcp.neon.tech/mcp, y creó neon.ts y el contexto de proyecto. No se agregó un segundo servidor MCP redundante.

Desde tu terminal local:

```powershell
npx --yes neon@latest login
npx --yes neon@latest init -y --org-id org-red-dew-78568003 --project-id silent-term-17758572 --branch production --services postgres
npx --yes neon@latest config plan
npx --yes neon@latest deploy
.\.venv\Scripts\python.exe deploy\setup_neon.py
```

Completa el login en el navegador que abra el CLI. Una sesión iniciada en tu laptop no autentica otro equipo o este entorno remoto. Si el despliegue no extrajo las variables, usa `npx --yes neon@latest env pull`. No envíes contraseñas o archivos .env al chat. No se habilitaron Auth, Functions, Object Storage, Data API ni AI Gateway: el servicio solicitado es Postgres y el proyecto conserva identidad Django y psycopg.

## Comprobaciones realizadas en este entorno

- `npm run build`: correcto; BUILD_OK: Django checks, Python compilation, JavaScript syntax and static assets.
- `npm test`: 26 pruebas correctas en SQLite de prueba aislado; no son evidencia de integración PostgreSQL.
- Servidor WSGI iniciado y probado por HTTP: /api/health 200 Django; / 200 HTML. Estos endpoints no consultan PostgreSQL.
- init: contexto de proyecto escrito offline, sin verificar acceso a organización/proyecto.
- deploy: Authentication timed out after 60 seconds.
- PostgreSQL: hostname directo y pooled no resolvieron en este entorno (Temporary failure in name resolution).
- verify_neon: falló con OperationalError. No se crearon tablas ni roles remotos ni se verificó un flujo de pacientes contra Neon. No hay despliegue productivo confirmado.

La URL compartida se guardó únicamente en configuración local privada y se excluye del paquete. Las conexiones de los ejemplos son marcadores, nunca contraseñas reales.
