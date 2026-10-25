# Aplicación móvil Android y iPhone

El mismo frontend se empaqueta dentro de una aplicación nativa con Capacitor 7. No es un APK/IPA ya firmado. Requiere backend HTTPS accesible desde el teléfono. No uses 127.0.0.1: en el teléfono significa el propio teléfono.

## Preparación

Node 22 LTS recomendado. En esta carpeta:

```bash
npm ci
```

En PowerShell:

```powershell
$env:POLISALUD_API_URL="https://tu-servidor.example"
npm run prepare:web
npx cap sync android
npx cap sync ios
```

En macOS/Linux: `export POLISALUD_API_URL=https://tu-servidor.example`.
Los proyectos android e ios ya están incluidos. Usa `cap add` únicamente si decides regenerar una plataforma desde cero. En Mac, instala CocoaPods y sincroniza iOS antes de abrir Xcode.

Android: instala Android Studio y sus SDK; `npm run android`. Desde Android Studio compila y firma el APK/AAB con tu propia clave.
iPhone: requiere macOS, Xcode y firma Apple; `npm run ios`. Configura el equipo de desarrollo y genera el archivo para dispositivo/TestFlight.

El backend admite los orígenes `https://localhost` (Android) y `capacitor://localhost` (iOS). No habilita CORS universal. El token nativo expira en 8 horas y se mantiene en memoria; no se guarda en localStorage. La API de pacientes aplica el mismo control de acceso que la web. La app requiere internet y no almacena expedientes para consulta offline.

Se incluye Bluetooth Low Energy con @capacitor-community/bluetooth-le 7.3.2 y protocolo GATT propio del firmware ESP32 incluido. El build prepara ble-native.js mediante esbuild y cap sync incorpora el plugin. iOS incluye NSBluetoothAlwaysUsageDescription. Para usarlo pulsa Prueba ESP32 → BLE en tu expediente. Ver ../docs/ESP32_MQ138.md. Requiere mantener la aplicación en primer plano y acceso al servidor Django para guardar datos; no se ofrece historial offline. Los proyectos se sincronizaron, pero no se compilaron ni probaron en teléfonos físicos.
