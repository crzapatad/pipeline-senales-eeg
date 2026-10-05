import numpy as np
import h5py
from scipy.signal import butter, filtfilt

def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    """Aplica filtro butterworth pasabanda"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, padlen=None)
    return filtered

# Cargar datos
print("Cargando datos EEG...")
with h5py.File("allChan_1kHz_clean.mat", "r") as f:
    matriz_eeg = np.array(f["allChan_clean"])
    Fs = 1000.0

print(f"Datos cargados: {matriz_eeg.shape}")
print(f"Frecuencia de muestreo: {Fs} Hz")
print()

# Definir canales a procesar
canales_cpf = list(range(1, 11))  # 1-10
canales_amy = list(range(17, 26))  # 17-25

# Crear mapeo de canales originales a columnas
n_canales = matriz_eeg.shape[1]
original_to_col = {i+1: i for i in range(n_canales)}

# Parámetros del filtro
filtro_low = 0.1  # Hz
filtro_high = 30.0  # Hz
order = 4

# Procesar canales CPF
print("Procesando canales CPF (1-10)...")
for canal in canales_cpf:
    col = original_to_col[canal]
    senal = matriz_eeg[:, col]
    
    # Aplicar filtro
    senal_filt = butterworth_bandpass(senal, Fs, filtro_low, filtro_high, order)
    
    # Guardar como .npy
    nombre_archivo = f"canal_{canal}_PFC_butterworth_0_30_n4.npy"
    np.save(nombre_archivo, senal_filt)
    print(f"  Guardado: {nombre_archivo}")

print()

# Procesar canales Amy
print("Procesando canales Amy (17-25)...")
for canal in canales_amy:
    col = original_to_col[canal]
    senal = matriz_eeg[:, col]
    
    # Aplicar filtro
    senal_filt = butterworth_bandpass(senal, Fs, filtro_low, filtro_high, order)
    
    # Guardar como .npy
    nombre_archivo = f"canal_{canal}_Amy_butterworth_0_30_n4.npy"
    np.save(nombre_archivo, senal_filt)
    print(f"  Guardado: {nombre_archivo}")

print()
print("=== PROCESAMIENTO COMPLETADO ===")
print(f"Se generaron {len(canales_cpf)} archivos para CPF")
print(f"Se generaron {len(canales_amy)} archivos para Amy")
print(f"Total: {len(canales_cpf) + len(canales_amy)} archivos .npy")
