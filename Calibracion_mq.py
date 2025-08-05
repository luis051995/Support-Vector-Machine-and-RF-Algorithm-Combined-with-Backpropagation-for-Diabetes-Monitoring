import numpy as np
import matplotlib.pyplot as plt

ppm = np.array([10, 20, 50, 100, 200, 400])
rs_r0 = np.array([4.5, 3.5, 2.3, 1.6, 1.1, 0.8])

print("===== Puntos aproximados =====")
print(" PPM \t Rs/R0 ")
for x, y in zip(ppm, rs_r0):
    print(f"{x:.1f} \t {y:.3f}")

log_ppm = np.log10(ppm)
log_rs_r0 = np.log10(rs_r0)

m, b = np.polyfit(log_ppm, log_rs_r0, 1)

print("\n===== Parámetros de ajuste (solo acetona) =====")
print(f" Pendiente (m): {m:.6f}")
print(f" Intersección (b): {b:.6f}")

x_fit = np.linspace(min(log_ppm), max(log_ppm), 100)
y_fit = m * x_fit + b


plt.figure(figsize=(8, 6))
plt.scatter(log_ppm, log_rs_r0, color='blue', label='Datos estimados acetona')
plt.plot(x_fit, y_fit, color='red', label=f'Ajuste: y = {m:.4f}x + {b:.4f}')
plt.xlabel('log(PPM acetona)')
plt.ylabel('log(Rs/R0)')
plt.title('Curva de calibración MQ-138 - Acetona')
plt.legend()
plt.grid(True)
plt.savefig('curva_acetona_loglog.png', dpi=300)
plt.show()

print("\n✅ Gráfica guardada: curva_acetona_loglog.png")
