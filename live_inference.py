import joblib
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model
from flask import Flask, jsonify
from flask_cors import CORS
import threading


app = Flask(__name__)
CORS(app) 

resultado_inferencia = {
    "ppm": None,
    "slope": None,
    "decision_score": None,
    "proba_svm": None,
    "nn_output": None,
    "riesgo": None
}

@app.route('/resultado', methods=['GET'])
def get_resultado():
    return jsonify(resultado_inferencia)


def iniciar_api():
    app.run(host='0.0.0.0', port=5000)

def main():
    global resultado_inferencia

    # Cargar modelos
    scaler = joblib.load('models/scaler.joblib')
    svm_model = joblib.load('models/svm_model.joblib')
    nn_model = load_model('models/nn_model.h5')
    print("✅ Modelos cargados correctamente.")

    # Parámetros del usuario
    edad = float(input("Edad (años): "))
    peso = float(input("Peso (kg): "))
    altura = float(input("Altura (cm): "))
    sexo = int(input("Sexo (0=Femenino, 1=Masculino): "))
    antecedentes = int(input("Antecedentes (0=No, 1=Sí): "))

    # Serial
    ser = serial.Serial('COM3', baudrate=9600, timeout=2)
    print("🔗 Conexión serial abierta...")

    # Calibración MQ-138
    RL = 10.0
    offset_samples = 30
    m = -0.4783
    b = 1.1537

    def read_voltage():
        raw = ser.readline().decode(errors='ignore').strip()
        try:
            v = float(raw)
            if v <= 0:
                return None
            return v
        except ValueError:
            return None

    print(f"🔧 Calibrando R0: tomando {offset_samples} muestras...")
    rs_values = []
    while len(rs_values) < offset_samples:
        v = read_voltage()
        if v is None:
            continue
        RS = ((3.3 * RL) / v) - RL
        rs_values.append(RS)
        time.sleep(0.2)

    R0 = float(np.mean(rs_values)) if rs_values else 10.0
    print(f"✅ Calibración completa. R0 = {R0:.2f} kΩ")

    baseline_measurements = 30
    print(f"🔧 Medición de línea base de PPM...")
    ppm_vals = []
    for _ in range(baseline_measurements):
        v = read_voltage()
        if v is None:
            continue
        RS = ((3.3 * RL) / v) - RL
        ratio = RS / R0
        ppm_vals.append(10 ** ((np.log10(ratio) - b) / m))
        time.sleep(0.2)

    baseline_ppm = float(np.mean(ppm_vals)) if ppm_vals else 0.0
    print(f"✅ Línea base de PPM = {baseline_ppm:.2f} ppm")

    prev_ppm = None
    slope_threshold = 1.0
    window_time = 1.0
    dt = window_time
    display_interval = 1.0
    last_display = time.time()

    print("Iniciando detección de soplo...")
    while True:
        try:
            v = read_voltage()
            if v is None:
                continue

            RS = ((3.3 * RL) / v) - RL
            ratio = RS / R0
            ppm = 10 ** ((np.log10(ratio) - b) / m)

            if prev_ppm is not None:
                slope = (ppm - prev_ppm) / dt
            else:
                slope = 0.0
            prev_ppm = ppm

            if time.time() - last_display >= display_interval:
                print(f"📡 PPM: {ppm:.2f} | Slope: {slope:.2f} ppm/s")
                last_display = time.time()

            if slope > slope_threshold:
                n_features = scaler.mean_.shape[0]
                input_features = np.zeros(n_features)
                input_features[0] = ppm
                input_features[1] = edad
                input_features[2] = sexo
                input_features[3] = antecedentes
                input_features[4] = peso
                input_features[5] = altura

                input_scaled = scaler.transform([input_features])
                decision_score = svm_model.decision_function(input_scaled)
                proba_svm = svm_model.predict_proba(input_scaled)[0][1]
                input_combined = np.column_stack((input_scaled, decision_score))
                nn_output = nn_model.predict(input_combined)[0][0]

                print(f"🔍 Score SVM: {decision_score[0]:.4f} | Prob SVM: {proba_svm:.4f} | NN salida: {nn_output:.4f}")
                print("🚨 Riesgo detectado.") if nn_output > 0.5 else print("✅ Nivel normal.")

                # 📡 Actualizar variable global para Ionic
                resultado_inferencia = {
                    "ppm": float(ppm),
                    "slope": float(slope),
                    "decision_score": float(decision_score[0]),
                    "proba_svm": float(proba_svm),
                    "nn_output": float(nn_output),
                    "riesgo": bool(nn_output > 0.5)
                }

            time.sleep(0.1)

        except Exception as e:
            print(f"⚠️ Error inesperado: {e}")


if __name__ == '__main__':
    # Hilo para la API
    api_thread = threading.Thread(target=iniciar_api)
    api_thread.daemon = True
    api_thread.start()

    # Script principal
    main()
