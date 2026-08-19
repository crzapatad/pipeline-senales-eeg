import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import time

print("Prueba de rendimiento de filtrado y Hilbert...")

# Cargar datos
cpf = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
amig = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
fs = 1000.0

# Extraer ventana pequeña
start_idx = int(1 * 60 * fs)
dur_samples = int(1 * 60 * fs)  # Solo 1 minuto
cpf_window = cpf[start_idx:start_idx + dur_samples]
amig_window = amig[start_idx:start_idx + dur_samples]

print(f"Ventana: {cpf_window.shape}")

# Prueba de filtrado para una frecuencia
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

start = time.time()
cpf_filtered = bandpass_filter(cpf_window, 4.0, 6.0, fs)
print(f"Filtrado CPF: {time.time() - start:.2f}s")

start = time.time()
cpf_phase = np.angle(hilbert(cpf_filtered))
print(f"Hilbert CPF: {time.time() - start:.2f}s")

print("Prueba completada.")
