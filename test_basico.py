import numpy as np
from scipy.signal import butter, filtfilt, hilbert

print("Script básico de prueba...")

# Simulación simple
cpf = np.random.randn(10000)
amig = np.random.randn(10000)

def filtro_pasabanda(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

print("Iniciando filtrado...")
cpf_filtered = filtro_pasabanda(cpf, 4.0, 6.0, 1000.0)
print("Filtrado completado")

print("Iniciando Hilbert...")
cpf_phase = np.angle(hilbert(cpf_filtered))
print("Hilbert completado")

print("Script básico completado exitosamente")
