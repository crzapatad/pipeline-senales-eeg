import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
import h5py

# Cargar datos
with h5py.File("allChan_1kHz_clean.mat", "r") as f:
    datos = np.array(f["allChan_clean"])

print("Datos cargados:", datos.shape)
print("Rango de datos:", datos.min(), datos.max())

# Tomar canales 1 y 17 (índices 0 y 16)
senal1 = datos[:, 0]  # Canal 1
senal2 = datos[:, 16]  # Canal 17

print("Señal 1 - rango:", senal1.min(), senal1.max(), "std:", senal1.std())
print("Señal 2 - rango:", senal2.min(), senal2.max(), "std:", senal2.std())

# Normalizar
senal1_norm = (senal1 - np.mean(senal1)) / (np.std(senal1) + 1e-10)
senal2_norm = (senal2 - np.mean(senal2)) / (np.std(senal2) + 1e-10)

print("Señal 1 normalizada - rango:", senal1_norm.min(), senal1_norm.max(), "std:", senal1_norm.std())
print("Señal 2 normalizada - rango:", senal2_norm.min(), senal2_norm.max(), "std:", senal2_norm.std())

# Ventana pequeña
window_size = 10000
senal1_win = senal1_norm[:window_size]
senal2_win = senal2_norm[:window_size]

print("Ventana - rango señal 1:", senal1_win.min(), senal1_win.max())
print("Ventana - rango señal 2:", senal2_win.min(), senal2_win.max())

# Filtro
fs = 1000.0
def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, padlen=None)
    return filtered

senal1_filt = butterworth_bandpass(senal1_win, fs, 0.1, 30.0, order=4)
senal2_filt = butterworth_bandpass(senal2_win, fs, 0.1, 30.0, order=4)

print("Después del filtro:")
print("Señal 1 filtrada - rango:", senal1_filt.min(), senal1_filt.max(), "std:", senal1_filt.std())
print("Señal 2 filtrada - rango:", senal2_filt.min(), senal2_filt.max(), "std:", senal2_filt.std())

# Correlación cruzada
cross_corr = correlate(senal1_filt, senal2_filt, mode='same', method='auto')
print("Correlación cruda - rango:", cross_corr.min(), cross_corr.max())

# Normalizar
n = len(senal1_filt)
max_lag_samples = 250
center_idx = len(cross_corr) // 2
start_idx = center_idx - max_lag_samples
end_idx = center_idx + max_lag_samples + 1

cross_corr_limited = cross_corr[start_idx:end_idx]
lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
normalization = n - np.abs(lags_samples)
normalization[normalization == 0] = 1

cross_corr_normalized = cross_corr_limited / normalization
print("Correlación normalizada - rango:", cross_corr_normalized.min(), cross_corr_normalized.max())

# Verificar si hay NaN o valores extraños
print("¿Hay NaN?", np.isnan(cross_corr_normalized).any())
print("¿Hay Inf?", np.isinf(cross_corr_normalized).any())