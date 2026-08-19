import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert
from itertools import combinations
from joblib import Parallel, delayed

def cargar_eeg(filepath):
    """Carga el archivo .mat (compatible con HDF5/v7.3)."""
    try:
        data_mat = sio.loadmat(filepath)
        keys = [k for k in data_mat.keys() if not k.startswith('__')]
        data = np.array(data_mat[keys[0]])
    except Exception:
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            data = np.array(f[keys[0]])
            
    # Asegurar dimensión (canales, tiempo)
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data

def calcular_mrl_frecuencia_offset(data, fs=500, freq_range=(1, 30), lag_max_sec=0.1, n_jobs=-1):
    n_canales, total_samples = data.shape
    pares = list(combinations(range(n_canales), 2))
    
    max_lag_samples = int(lag_max_sec * fs)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = (lags / fs) * 1000  # Convertir a milisegundos
    
    frecuencias = np.arange(freq_range[0], freq_range[1] + 1)
    heatmap_mrl = np.zeros((len(lags), len(frecuencias)))

    print(f"Calculando MRL para {len(frecuencias)} frecuencias y {len(lags)} offsets...")

    for f_idx, freq in enumerate(frecuencias):
        print(f" -> Procesando Frecuencia: {freq} Hz...")
        
        # Filtro pasa-banda para la frecuencia actual (+/- 1 Hz)
        f_low = max(0.5, freq - 1)
        f_high = freq + 1
        sos = butter(3, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
        
        # Filtrar señal y extraer fase instantánea por Hilbert
        eeg_filtrado = sosfiltfilt(sos, data, axis=1)
        fases = np.angle(hilbert(eeg_filtrado, axis=1))

        # Calcular MRL promedio sobre todos los pares para esta frecuencia
        def _mrl_par(par):
            ch1, ch2 = par
            f1, f2 = fases[ch1], fases[ch2]
            n = len(f1)
            res = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                if lag < 0:
                    d = np.exp(1j * (f1[-lag:] - f2[:n + lag]))
                elif lag > 0:
                    d = np.exp(1j * (f1[:n - lag] - f2[lag:]))
                else:
                    d = np.exp(1j * (f1 - f2))
                res[i] = np.abs(np.mean(d))
            return res

        mrl_pares = Parallel(n_jobs=n_jobs)(delayed(_mrl_par)(p) for p in pares)
        heatmap_mrl[:, f_idx] = np.mean(mrl_pares, axis=0)

    return frecuencias, lags_ms, heatmap_mrl

# ==============================================================================
# EJECUCIÓN Y GRÁFICO
# ==============================================================================
if __name__ == '__main__':
    archivo = "allChan_phase_instantanea.mat"
    fs = 500
    
    # Cargar y procesar
    data = cargar_eeg(archivo)
    freqs, lags_ms, mapa_mrl = calcular_mrl_frecuencia_offset(
        data, fs=fs, freq_range=(1, 30), lag_max_sec=0.1
    )

    # Graficar Heatmap 2D
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(freqs, lags_ms, mapa_mrl, shading='gouraud', cmap='viridis')
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('Mean Resultant Length (MRL)', fontsize=11)

    plt.axhline(0, color='white', linestyle='--', alpha=0.7, label='Offset 0 ms')
    plt.title('Mapa de MRL: Frecuencia vs. Offset (Lag)', fontsize=13)
    plt.xlabel('Frecuencia (Hz)', fontsize=11)
    plt.ylabel('Offset / Lag (ms)', fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.show()