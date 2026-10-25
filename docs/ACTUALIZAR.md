# Actualizar sin perder expedientes

Esta actualización conserva el autorregistro y añade pruebas individuales con ESP32/MQ-138 por Wi-Fi o Bluetooth LE. La administración global sigue siendo exclusiva del administrador.

1. Detén el servidor con Ctrl+C.
2. Haz una copia segura de tu carpeta actual. Conserva especialmente `.env` e `instance.sqlite3` si trabajas con SQLite local. No envíes esos archivos por chat. Si usas PostgreSQL, respalda esa base y las claves de cifrado por separado.
3. Descomprime el ZIP nuevo en otra carpeta. Copia el contenido de `polisalud_django` sobre los archivos de tu proyecto actual. El ZIP no contiene `.env` ni una base de pacientes, para no reemplazar los tuyos. No borres los originales ni generes una clave de cifrado nueva.
4. Desde PowerShell, ubicado donde está `manage.py`, ejecuta:

```powershell
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py collectstatic --noinput
$env:DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1,192.168.100.28"
.\.venv\Scripts\python.exe manage.py runserver 0.0.0.0:8000
```

Las migraciones añaden el límite de registros y las tablas de sesiones, calibraciones y muestras del sensor; no borran expedientes ni usuarios. La IP corresponde a la conexión compartida por el usuario; si cambia, ajusta hosts e IP del navegador.

5. Recarga el navegador. En teléfono, entra a http://192.168.100.28:8000 desde el mismo Wi-Fi. Pulsa **Crear mi cuenta de paciente**. Inicia sesión después del registro.

Cada paciente abre directamente su propio historial y no puede ver el directorio global. El administrador conserva acceso a todos los expedientes, consultas, mediciones, usuarios y auditoría. Un rol profesional heredado ya no puede consultar todos los expedientes.

Las altas crean siempre un expediente nuevo. No vinculan automáticamente una cuenta a un historial previo por nombre/teléfono. Si ya existía el expediente, el administrador puede crear la cuenta vinculada desde Usuarios, evitando duplicarlo.

La prueba local HTTP se limita a datos ficticios. Para publicar registros reales se requiere HTTPS y el despliegue apropiado descrito en la documentación.

## ESP32/MQ-138

En cada expediente aparece **Prueba ESP32** y una pestaña **Pruebas ESP32**. Lee ESP32_MQ138.md antes de cablear. En local, Wi-Fi funciona con el servidor compartido por la laptop; BLE desde web requiere navegador compatible y contexto seguro, o la app nativa compilada. El firmware está en firmware/esp32_mq138.
