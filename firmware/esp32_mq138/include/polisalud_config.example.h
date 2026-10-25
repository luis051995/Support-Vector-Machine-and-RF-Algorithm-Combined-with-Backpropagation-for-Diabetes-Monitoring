#pragma once
// Copia como polisalud_config.h y configura localmente. No subas polisalud_config.h al repositorio.
#define WIFI_SSID "TU_WIFI_2_4_GHZ"
#define WIFI_PASSWORD "TU_CLAVE_WIFI"
#define SENSOR_PIN 34  // ADC1 del ESP32 clásico (WROOM); no usar ADC2 con Wi-Fi.
#define PAIR_BUTTON 0  // BOOT: pulsar después del arranque, no al energizar.
// Elige un PIN único de 6 dígitos entre 100000 y 999999. 0 deshabilita BLE.
#define BLE_PAIR_PIN 0
#define ALLOW_LAN_HTTP false // true SOLO laboratorio con datos ficticios; admite RFC1918.
// CA raíz válida del servidor HTTPS. No se omite validación de certificado.
static const char ROOT_CA_PEM[] = "";
