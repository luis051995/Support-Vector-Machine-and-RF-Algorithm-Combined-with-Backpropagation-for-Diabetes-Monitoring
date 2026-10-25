# Verificación · 24 de octubre de 2026

- 26 pruebas aprobadas con `python manage.py test tests` usando SQLite de prueba y cifrado real de payloads.
- Se comprobó con SQL directo que nombre y notas no aparecen en claro; lectura con clave incorrecta falla cerrada.
- Pruebas de control de acceso, expedientes aislados por paciente, CSRF en login y mutaciones, exportación auditada, inferencia persistida y validación de entradas.
- Rechazo de origen móvil no autorizado, expiración de token, invalidación de sesiones por cambio de contraseña, bloqueo de intentos con django-axes y rutas de secretos no públicas.
- El dominio y la aplicación se importan usando Python `-S`, sin dependencias externas; un caso de uso funciona con repositorio e IA en memoria.
- Migraciones generadas y aplicadas localmente; `makemigrations --check --dry-run` no detecta cambios pendientes.
- Dependencias instaladas desde requirements.txt; sintaxis JS y compilación Python correctas; estáticos recogidos con collectstatic.
- `check --deploy --fail-level WARNING` pasó con variables de producción ficticias; esa comprobación no establece conexión con PostgreSQL.
- No se dispuso de una base PostgreSQL remota ni servidor local PostgreSQL. La negociación TLS y los GRANT/REVOKE están preparados pero no fueron ejecutados contra Neon.
- No se publicó el backend Django en Render ni se aprovisionó una base Neon: las integraciones están disponibles pero sin vincular a cuentas del usuario.
- La interfaz web se conserva de la entrega anterior, ajustando obtención y cabecera CSRF al contrato Django; esta iteración no hizo una revisión visual adicional en navegador.
- Recursos web móviles actualizados; no se compilaron ni firmaron APK/IPA ni se probaron teléfonos. No se conectó un sensor físico.
- El modelo y sus métricas se conservaron; no se volvió a entrenar por el cambio de framework. No se realizó una auditoría externa de seguridad ni validación clínica.

Actualización de registro: se verificó alta con cuenta y expediente cifrado, rol paciente fijado por servidor aun cuando el cliente solicita administrador, reversión ante usuario duplicado, consentimiento, CSRF, límite de registros y restricción de acceso global al antiguo rol profesional.


## Integración ESP32/MQ-138

- 26 pruebas en total aprobadas, incluidas ocho de captura: separación por paciente, tokens de dispositivo, caducidad/cancelación, CSRF BLE, transporte permitido, secuencia/idempotencia, límite de muestras, calibración solo por administrador, señal cifrada y resumen de inferencia sin duplicados.
- El firmware compiló con PlatformIO para esp32dev, Arduino core 2.0.17 y flash de 4 MB con partición huge_app. RAM estática 58,864 bytes; imagen 1,658,425 bytes. Compilación con configuración de ejemplo, sin claves de Wi-Fi y BLE deshabilitado hasta configurar PIN único.
- Recursos móviles preparados y plugin @capacitor-community/bluetooth-le 7.3.2 sincronizado para Android/iOS. Esbuild generó el adaptador BLE nativo. CocoaPods/Xcode no disponibles: no se compiló ni firmó una app nativa.
- El protocolo binario de 16 bytes coincide entre firmware y decodificador web/nativo.
- No se conectó un ESP32 físico, no se probó radio Wi-Fi/BLE en teléfonos ni se verificó el montaje eléctrico, calentamiento, calibración, precisión o selectividad del MQ-138.
- No se cargó el firmware a una placa del usuario ni se verificó visualmente la nueva pantalla de pruebas en un navegador. Se comprobó sintaxis JavaScript y endpoints automatizados.
- Wi-Fi necesita provisionar URL/token por monitor serial para cada sesión. BLE requiere emparejamiento físico y cliente compatible; la app móvil sigue requiriendo firma/instalación. Ambos necesitan Django accesible para persistir.

## Preparación Neon

npm run build pasó. npm test pasó 26 pruebas aisladas de SQLite. El servidor respondió HTTP 200 en raíz y health. La integración real con PostgreSQL no se completó: resolución DNS fallida y autenticación de neon deploy expirada. Consulta CONECTAR_NEON.md.
