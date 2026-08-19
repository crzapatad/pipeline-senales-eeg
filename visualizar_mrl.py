import matplotlib.pyplot as plt
import scipy.io as sio

# 1. Cargar el archivo generado por el script anterior
data = sio.loadmat('resultado_mrl.mat')
lags_ms = data['lags_sec'].flatten() * 1000  # Convertir a milisegundos

plt.figure(figsize=(10, 5))

# 2. Graficar el promedio de MRL sobre todos los pares para cada ventana
for key in data.keys():
    if not key.startswith('__') and key not in ['lags_sec', 'pares']:
        mrl_matriz = data[key]  # Estructura: (pares, lags)
        mrl_promedio = mrl_matriz.mean(axis=0)  # Promedio de todos los pares
        
        # Resaltar la ventana Real con una línea más gruesa
        lw = 2.5 if key == 'Real' else 1.5
        plt.plot(lags_ms, mrl_promedio, label=f"Ventana {key}", linewidth=lw)

# 3. Formato del gráfico
plt.title("Comparación de MRL Promedio por Ventana de Tiempo", fontsize=14)
plt.xlabel("Lag (ms)", fontsize=12)
plt.ylabel("Mean Resultant Length (MRL)", fontsize=12)
plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Lag 0 ms')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

# 4. Mostrar gráfico en pantalla
plt.show()