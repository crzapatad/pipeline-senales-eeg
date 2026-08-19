import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import time

print("Iniciando prueba simplificada...")

# Cargar datos
cpf = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
amig = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
fs = 1000.0

print(f"Datos cargados: CPF={cpf.shape}, Amígdala={amig.shape}")

# Probar filtrado y Hilbert en una ventana pequeña
start = time.time()
dur_min = 5
dur_samples = int(dur_min * 60 * fs)
start_idx = int(1 * 60 * fs)  # minuto 1

cpf_window = cpf[start_idx:start_idx + dur_samples]
amig_window = amig[start_idx:start_idx + dur_samples]

print(f"Ventana extraída: {cpf_window.shape}")

# Filtro pasabanda
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Probar filtrado en una frecuencia
f_low, f_high = 4.0, 6.0  # 5 Hz +/- 1 Hz
start = time.time()
cpf_filtered = bandpass_filter(cpf_window, f_low, f_high, fs)
print(f"Filtrado CPF: {cpf_filtered.shape}, tiempo: {time.time() - start:.2f}s")

start = time.time()
amig_filtered = bandpass_filter(amig_window, f_low, f_high, fs)
print(f"Filtrado Amígdala: {amig_filtered.shape}, tiempo: {time.time() - start:.2f}s")

# Probar Hilbert
start = time.time()
cpf_analytic = hilbert(cpf_filtered)
cpf_phase = np.angle(cpf_analytic)
print(f"Hilbert CPF: {cpf_phase.shape}, tiempo: {time.time() - start:.2f}s")

start = time.time()
amig_analytic = hilbert(amig_filtered)
amig_phase = np.angle(amig_analytic)
print(f"Hilbert Amígdala: {amig_phase.shape}, tiempo: {time.time() - start:.2f}s")

print("Prueba completada exitosamente.")
