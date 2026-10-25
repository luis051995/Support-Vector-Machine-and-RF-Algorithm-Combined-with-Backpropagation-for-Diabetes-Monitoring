# PoliSalud · Django con arquitectura hexagonal

Propuesta académica para el IPN de Mtro. José Luis Robles Bárcenas; sin afiliación o aprobación institucional oficial. Esta edición sustituye el backend Flask por Django 5.2 LTS, Django REST Framework y PostgreSQL para el despliegue. Se mantienen la web, el modelo de investigación y los proyectos fuente Android/iOS. Es un prototipo funcional, no un sistema clínico certificado.

## Inicio local en Windows

Instala Python 3.12 de 64 bits. Descomprime y ejecuta INICIAR_WINDOWS.bat. Crea el administrador cuando se solicite y abre http://127.0.0.1:8000. No se muestran los caracteres al escribir la contraseña. La contraseña requiere 12 caracteres y no debe ser común.

Alternativa manual:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe setup_local.py
.venv\Scripts\python.exe manage.py migrate
.venv\Scripts\python.exe manage.py collectstatic --noinput
.venv\Scripts\python.exe manage.py createsuperuser
.venv\Scripts\python.exe run_local.py
```

En Linux/macOS activa un entorno virtual y usa los mismos comandos con `python`. La base local se crea automáticamente como instance.sqlite3 y sus payloads clínicos también se cifran. `setup_local.py` crea .env una sola vez y no sobrescribe claves existentes. Respáldalo de forma segura y nunca lo subas a Git.

## Funcionalidades

- Alta y edición de pacientes, consultas, notas, plan, mediciones y evolución individual.
- Registro público de cuentas paciente con expediente individual; cada paciente solo lee/exporta su propio historial. Solo el administrador consulta y administra el conjunto de expedientes.
- Datos personales y clínicos cifrados antes de almacenarse; contraseñas con hash Argon2.
- Auditoría por caso de uso y protección de entradas históricas.
- Sesiones web Django con CSRF y tokens móviles revocables con caducidad.
- Modelo SVM + RF + MLP ya entrenado, inferencia con orden de variables explícito y métricas conservadas.
- PostgreSQL con verificación TLS, credenciales separadas para migraciones y ejecución, y permisos limitados.
- render.yaml y scripts para Render Free + Neon Free.

## Arquitectura

| Ruta | Capa |
|---|---|
| clinical/domain/ | Entidades y contratos puros |
| clinical/application/ | Casos de uso independientes del framework |
| clinical/adapters/http/ | Entrada Django REST |
| clinical/adapters/persistence/ | ORM, cifrado, transacciones, migraciones |
| clinical/adapters/ml/ | Inferencia scikit-learn |
| clinical/bootstrap.py | Composición e inyección de dependencias |
| identity/ | Usuarios y sesiones Django |
| config/ | Configuración y seguridad HTTP |
| deploy/ | Hosting y privilegios PostgreSQL |
| web/, mobile/ | Clientes web, Android y iOS |

Ver docs/ARQUITECTURA.md, docs/SEGURIDAD.md y docs/SERVIDOR_GRATIS.md.

## Pruebas y modelo

```bash
python manage.py test tests
python manage.py check
python -m ml.train
```

Para `check --deploy` configura primero las variables de producción descritas en la guía. Nunca uses la clave de desarrollo en un hosting público. El algoritmo mantiene los resultados internos de la versión anterior; el cambio de framework no demuestra una mejora de desempeño clínico. Modelo y dataset conservados son de investigación. Se mantienen sus limitaciones de procedencia y ausencia de identificación real por paciente.

## Entrega móvil y hospedaje

Los proyectos Android/iOS se incluyen como fuente, sin APK/IPA firmado. Requieren apuntar al servidor Django HTTPS y compilar/sincronizar con las herramientas respectivas. No se realizó publicación en tiendas ni aprovisionamiento en Render/Neon. La guía permite preparar el despliegue en cuentas propias; los planes gratuitos no garantizan continuidad ni adecuación para expedientes reales.

No se importan automáticamente usuarios o expedientes del anterior SQLite de Flask. Esta versión inicializa una base nueva; si ya tienes datos reales, conserva la instalación anterior y planifica una migración validada antes de reemplazarla. No se borraron los archivos de la versión anterior.

Licencia GPL-3.0 conservada del proyecto base; autoría en CITATION.cff.

## Registro de pacientes

En la pantalla inicial pulsa **Crear mi cuenta de paciente**. El alta solicita datos personales, usuario, contraseña y consentimiento; crea cuenta y expediente en la misma transacción. Después inicia sesión. El usuario nuevo siempre es paciente y no puede elegir un expediente existente, un rol administrador ni permisos. Solo el administrador registra consultas, mediciones y edita expedientes. El rol heredado profesional ya no tiene acceso global.

Ver docs/ACTUALIZAR.md antes de reemplazar una instalación existente.

## Prueba en tiempo real con ESP32 + MQ-138

Cada paciente puede iniciar una prueba propia desde **Prueba ESP32**. El ESP32 transmite por Wi-Fi a Django o por BLE al cliente que retransmite a Django. Señal cruda/voltaje en vivo, historial cifrado de pruebas, tokens temporales y límites de muestras. Solo el administrador registra calibraciones de acetona; sin ellas se conserva señal cruda y no se ejecuta IA. Firmware para ESP32 clásico y guía de cableado/provisión en docs/ESP32_MQ138.md. No se probó físicamente la placa ni el sensor del usuario. Los APK/IPA siguen requiriendo compilación y firma.

## Neon CLI e integración

Esta edición incorpora neon.ts, npm run build y deploy/setup_neon.py. Sigue docs/CONECTAR_NEON.md para usar la conexión de tu proyecto existente. Django utiliza APP_DATABASE_URL para el rol limitado y conserva DATABASE_URL/DATABASE_URL_UNPOOLED como variables administradas por Neon. La verificación remota no pudo completarse por autenticación CLI y resolución DNS en este entorno; no se afirma un despliegue productivo.
