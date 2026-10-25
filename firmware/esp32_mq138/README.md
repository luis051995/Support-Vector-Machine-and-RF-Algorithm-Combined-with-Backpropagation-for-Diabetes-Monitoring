# Firmware ESP32 clásico / MQ-138

Lee ../../docs/ESP32_MQ138.md antes de conectar. Copia include/polisalud_config.example.h como polisalud_config.h; configura Wi-Fi y PIN único de BLE. No se incluyen secretos reales. El esquema de ejemplo asume módulo MQ-138 AO alimentado a 5 V, con divisor de tensión hasta ADC1 GPIO34; no conecta 5 V al GPIO.

PlatformIO: platform espressif32 6.10.0, Arduino core 2.0.17, board esp32dev, ArduinoJson 7.4.2. Usa partición huge_app en flash de 4 MB y no habilita OTA. Ajustar placa/partición antes de usar variantes distintas.

Wi-Fi: acepta por serial un JSON de sesión con url, token y mode wifi, generado por Django. BLE: pulsar BOOT tras arrancar anuncia por 60 s; usa PIN y cifrado autenticado. Muestra: uint32 seq, uint32 uptime_ms, uint16 raw_adc, uint16 adc_mv, uint32 protocol=1, little-endian (16 bytes). No envía nombres ni diagnóstico.

Compilación/carga: python -m platformio run; python -m platformio run --target upload. Se requiere el toolchain oficial y acceso USB a tu placa. No interpretes una compilación correcta como validación del hardware ni de la medición.
