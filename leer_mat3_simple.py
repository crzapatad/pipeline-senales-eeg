import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# ==========================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Aplica un filtro Butterworth pasabanda a la señal."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_phase_signal(signal, lowcut, highcut, fs):
    """Filtra la señal en la banda deseada y extrae la fase instantánea mediante Hilbert."""
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    analytic = hilbert(filtered)
    return np.angle(analytic)

def compute_mrl_with_lags(phase1, phase2, lags_samples):
    """Calcula el MRL para cada lag entre dos fases."""
    N = len(phase1)
    mrls = []
    
    for lag in lags_samples:
        if lag < 0:
            s1 = phase1[-lag:]
            s2 = phase2[:N + lag]
        elif lag > 0:
            s1 = phase1[:N - lag]
            s2 = phase2[lag:]
        else:
            s1 = phase1
            s2 = phase2
            
        phase_diff = np.exp(1j * (s1 - s2))
        mrl = np.abs(np.mean(phase_diff))
        mrls.append(mrl)
    
    return np.array(mrls)

def analyze_frequency_band(signal_a, signal_b, fs, start_s, end_s, freq_center, bandwidth_hz, max_lag_ms):
    """Analiza una banda de frecuencia específica."""
    # Extraer ventana de tiempo
    start_idx = int(start_s * fs)
    end_idx = int(end_s * fs)
    
    window_a = signal_a[start_idx:end_idx]
    window_b = signal_b[start_idx:end_idx]
    
    # Asegurar misma longitud
    min_len = min(len(window_a), len(window_b))
    window_a = window_a[:min_len]
    window_b = window_b[:min_len]
    
    # Calcular fases
    f_low = max(0.1, freq_center - bandwidth_hz/2)
    f_high = freq_center + bandwidth_hz/2
    
    print(f"  Procesando frecuencia {freq_center} Hz (rango: {f_low:.1f}-{f_high:.1f} Hz)...")
    
    phase_a = get_phase_signal(window_a, f_low, f_high, fs)
    phase_b = get_phase_signal(window_b, f_low, f_high, fs)
    
    # Calcular MRL con diferentes lags
    max_lag_sec = max_lag_ms / 1000.0
    max_lag_samples = int(max_lag_sec * fs)
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = (lags_samples / fs) * 1000.0
    
    mrls = compute_mrl_with_lags(phase_a, phase_b, lags_samples)
    
    # Encontrar el lag donde MRL es máximo
    max_idx = np.argmax(mrls)
    optimal_lag = lags_ms[max_idx]
    max_mrl = mrls[max_idx]
    
    return {
        'frequency': freq_center,
        'lags_ms': lags_ms,
        'mrls': mrls,
        'optimal_lag_ms': optimal_lag,
        'max_mrl': max_mrl
    }

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("=== ANÁLISIS MRL SIMPLIFICADO ===")
    
    # Cargar datos
    print("Cargando allChan_phase_instantanea.mat...")
    with h5py.File("allChan_phase_instantanea.mat", "r") as f:
        signal_data = np.array(f["allChan_phase_instantanea"])
    
    # Transponer para tener (canales, muestras)
    signal_data = signal_data.T  # (30, 1284335)
    
    fs = 500  # Frecuencia de muestreo
    
    print(f"Datos cargados: {signal_data.shape[0]} canales, {signal_data.shape[1]} muestras")
    print(f"Duración total: {signal_data.shape[1]/fs/60:.2f} minutos")
    
    # Seleccionar canales (ajusta según tus necesidades)
    signal_a = signal_data[0, :]  # Canal 1
    signal_b = signal_data[16, :] if signal_data.shape[0] > 16 else signal_data[1, :]  # Canal 17
    
    print(f"Canales seleccionados: A shape={signal_a.shape}, B shape={signal_b.shape}")
    
    # Parámetros de análisis
    start_min = 1
    end_min = 6
    start_s = start_min * 60
    end_s = end_min * 60
    
    freq_min = 1
    freq_max = 10  # Reducido para ejecución más rápida
    freq_step = 1
    bandwidth_hz = 2
    max_lag_ms = 100  # Reducido para ejecución más rápida
    
    print(f"\nParámetros:")
    print(f"  Ventana de tiempo: {start_min}-{end_min} minutos")
    print(f"  Rango de frecuencias: {freq_min}-{freq_max} Hz (paso: {freq_step} Hz)")
    print(f"  Ancho de banda: {bandwidth_hz} Hz")
    print(f"  Rango de lags: ±{max_lag_ms} ms")
    
    # Análisis por frecuencia
    frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
    results = []
    
    for freq in frequencies:
        result = analyze_frequency_band(
            signal_a, signal_b, fs, start_s, end_s, 
            freq, bandwidth_hz, max_lag_ms
        )
        results.append(result)
        print(f"    -> Lag óptimo: {result['optimal_lag_ms']:.2f} ms, MRL máximo: {result['max_mrl']:.4f}")
    
    # Resumen de resultados
    print(f"\n=== RESUMEN DE RESULTADOS ===")
    optimal_lags = [r['optimal_lag_ms'] for r in results]
    max_mrls = [r['max_mrl'] for r in results]
    
    print(f"Frecuencias analizadas: {frequencies}")
    print(f"Lags óptimos (ms): {optimal_lags}")
    print(f"MRL máximos: {max_mrls}")
    
    # Gráfico de frecuencia vs lag óptimo
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, optimal_lags, 'o-', linewidth=2, markersize=8)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Lag óptimo (ms)')
    plt.title('Lag óptimo vs Frecuencia')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, max_mrls, 's-', color='orange', linewidth=2, markersize=8)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('MRL máximo')
    plt.title('MRL máximo vs Frecuencia')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mrl_analysis_simple.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado: mrl_analysis_simple.png")
    
    # Gráfico de MRL vs lag para cada frecuencia
    n_freqs = len(results)
    cols = 3
    rows = (n_freqs + cols - 1) // cols  # Calcular filas necesarias
    
    plt.figure(figsize=(15, 5 * rows))
    for i, result in enumerate(results):
        plt.subplot(rows, cols, i+1)
        plt.plot(result['lags_ms'], result['mrls'], linewidth=1.5)
        plt.axvline(result['optimal_lag_ms'], color='red', linestyle='--', alpha=0.7)
        plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
        plt.title(f"{result['frequency']} Hz")
        plt.xlabel('Lag (ms)')
        plt.ylabel('MRL')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mrl_by_frequency.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado: mrl_by_frequency.png")
    
    print("\n=== ANÁLISIS COMPLETADO ===")