import serial
import time
import numpy as np


SERIAL_PORT = 'COM3'  
BAUDRATE = 9600

RL = 10.0

m = -0.478285  
b = 1.153733   
R0 = 10.0  

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2)
print(f"🔗 Puerto serial abierto: {SERIAL_PORT} (Baudrate: {BAUDRATE})")


def read_voltage():
    line = ser.readline().decode(errors='ignore').strip()
    try:
        v = float(line)
        if v <= 0:
            return None
        return v
    except:
        return None

print("📡 Iniciando lectura continua de PPM (Ctrl+C para detener)")

try:
    while True:
        v = read_voltage()
        if v is None:
            continue
        
        RS = ((3.3 * RL) / v) - RL

        ratio = RS / R0

        ppm = 10 ** ((np.log10(ratio) - b) / m)

        print(f"PPM acetona: {ppm:.2f}")

        time.sleep(1.0)  
except KeyboardInterrupt:
    print("\n⏹️ Lectura detenida por el usuario.")

finally:
    ser.close()
    print("🔌 Puerto serial cerrado.")
