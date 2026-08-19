import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Iniciando prueba mínima...")

# Cargar datos
cpf = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
amig = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
fs = 1000.0

print(f"Datos cargados: CPF={cpf.shape}, Amígdala={amig.shape}")

# Parámetros mínimos
start_min = 1
dur_min = 5
freq_center = 5  # Solo una frecuencia
f_low = 4.0
f_high = 6.0
max_lag_ms = 100
max_lag_sec = max_lag_ms / 1000.0
max_lag_samples = int(max_lag_sec * fs)

# Extraer ventana
start_idx = int(start_min * 60 * fs)
dur_samples = int(dur_min * 60 * fs)

cpf_window = cpf[start_idx:start_idx + dur_samples]
amig_window = amig[start_idx:start_idx + dur_samples]

print(f"Ventana extraída: {cpf_window.shape}")

# Filtro
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

cpf_filtered = bandpass_filter(cpf_window, f_low, f_high, fs)
amig_filtered = bandpass_filter(amig_window, f_low, f_high, fs)

# Hilbert
cpf_phase = np.angle(hilbert(cpf_filtered))
amig_phase = np.angle(hilbert(amig_filtered))

print(f"Fases calculadas: CPF={cpf_phase.shape}, Amígdala={amig_phase.shape}")

# Calcular MRL para un solo lag
lags_samples = np.array([0])
lags_ms = np.array([0.0])

N = len(cpf_phase)
mrl = np.abs(np.mean(np.exp(1j * (amig_phase - cpf_phase))))
print(f"MRL en lag 0: {mrl:.4f}")

print("Prueba mínima completada exitosamente.")
