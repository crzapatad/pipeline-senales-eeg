import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from joblib import Parallel, delayed
import time

# ==========================================
# CONFIGURACIÓN OPTIMIZADA
# ==========================================

# Parámetros conservadores para velocidad
FREQ_MIN = 1           # Frecuencia mínima (Hz)
FREQ_MAX = 10          # Frecuencia máxima (Hz) - reducido de 30 para velocidad
FREQ_STEP = 1          # Paso de frecuencia (Hz)
BANDWIDTH_HZ = 2       # Ancho de banda (Hz)
MAX_LAG_MS = 100       # Rango de lags (ms) - reducido de 250 para velocidad
FS = 500               # Frecuencia de muestreo (Hz)
WINDOW_MIN = 5         # Duración de ventana (minutos)
# N_RANDOM_WINDOWS eliminado - ya no comparamos ventanas aleatorias

# ==========================================
# FUNCIONES OPTIMIZADAS
# ==========================================

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Filtro Butterworth pasabanda optimizado."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_phase_signal(signal, lowcut, highcut, fs):
    """Extracción de fase instantánea optimizada."""
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    analytic = hilbert(filtered)
    return np.angle(analytic)

def compute_mrl_single_lag(phase1, phase2, lag):
    """Cálculo MRL para un solo lag (optimizado para paralelización)."""
    N = len(phase1)
    if lag < 0:
        s1 = phase1[-lag:]
        s2 = phase2[:N + lag]
    elif lag > 0:
        s1 = phase1[:N - lag]
        s2 = phase2[lag:]
    else:
        s1 = phase1
        s2 = phase2
    
    # Usar cálculo vectorizado
    phase_diff = s1 - s2
    mrl = np.abs(np.mean(np.exp(1j * phase_diff)))
    return mrl

def compute_mrl_with_lags_parallel(phase1, phase2, lags_samples, n_jobs=-1):
    """Cálculo MRL con paralelización."""
    N = len(phase1)
    
    # Verificar si vale la pena paralelizar
    if len(lags_samples) < 10:
        # Para pocos lags, usar secuencial
        mrls = [compute_mrl_single_lag(phase1, phase2, lag) for lag in lags_samples]
        return np.array(mrls)
    else:
        # Para muchos lags, usar paralelización
        mrls = Parallel(n_jobs=n_jobs)(
            delayed(compute_mrl_single_lag)(phase1, phase2, lag) 
            for lag in lags_samples
        )
        return np.array(mrls)

def analyze_frequency_window(signal_a, signal_b, fs, start_s, end_s, freq_center, bandwidth_hz, max_lag_ms):
    """Análisis optimizado para una frecuencia específica."""
    # Extraer ventana
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
    
    phase_a = get_phase_signal(window_a, f_low, f_high, fs)
    phase_b = get_phase_signal(window_b, f_low, f_high, fs)
    
    # Calcular MRL con lags
    max_lag_sec = max_lag_ms / 1000.0
    max_lag_samples = int(max_lag_sec * fs)
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = (lags_samples / fs) * 1000.0
    
    mrls = compute_mrl_with_lags_parallel(phase_a, phase_b, lags_samples)
    
    # Encontrar máximo
    max_idx = np.argmax(mrls)
    optimal_lag = lags_ms[max_idx]
    max_mrl = mrls[max_idx]
    
    return {
        'frequency': freq_center,
        'optimal_lag_ms': optimal_lag,
        'max_mrl': max_mrl,
        'lags_ms': lags_ms,
        'mrls': mrls
    }

def analyze_window_optimized(signal_a, signal_b, fs, start_s, end_s, window_name, frequencies, bandwidth_hz, max_lag_ms):
    """Análisis optimizado de una ventana completa."""
    print(f"Analizando ventana {window_name} ({start_s/60:.1f}-{end_s/60:.1f} min)...")
    start_time = time.time()
    
    results = []
    for i, freq in enumerate(frequencies):
        if i % 5 == 0:  # Progreso cada 5 frecuencias
            elapsed = time.time() - start_time
            print(f"  Progreso: {i+1}/{len(frequencies)} frecuencias ({elapsed:.1f}s)")
        
        result = analyze_frequency_window(
            signal_a, signal_b, fs, start_s, end_s, 
            freq, bandwidth_hz, max_lag_ms
        )
        result['window'] = window_name
        results.append(result)
    
    elapsed = time.time() - start_time
    print(f"  Ventana {window_name} completada en {elapsed:.1f}s")
    
    return results

# ==========================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ==========================================

def save_lagged_mrl_optimized(signal_a, signal_b, fs, window_name, real_start_s, real_end_s):
    """
    Versión optimizada del análisis MRL - SOLO VENTANA REAL.
    - Elimina completamente la comparación con ventanas aleatorias
    - Reduce rango de frecuencias para velocidad
    - Usa paralelización donde posible
    - Análisis directo y eficiente
    """
    
    print("=" * 60)
    print("ANÁLISIS MRL OPTIMIZADO - SOLO VENTANA REAL")
    print("=" * 60)
    
    # Configuración optimizada
    frequencies = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)
    window_s = WINDOW_MIN * 60  # 5 minutos en segundos
    
    print(f"Parámetros optimizados:")
    print(f"  Rango frecuencias: {FREQ_MIN}-{FREQ_MAX} Hz (paso: {FREQ_STEP} Hz)")
    print(f"  Rango de lags: ±{MAX_LAG_MS} ms")
    print(f"  Ventana real: {real_start_s/60:.1f}-{real_end_s/60:.1f} min")
    print(f"  Sin comparación de ventanas aleatorias (máxima velocidad)")
    
    # Verificar duración suficiente
    total_duration = len(signal_a) / fs
    if total_duration < window_s:
        raise ValueError(f"Señal too short: {total_duration/60:.1f} min < {WINDOW_MIN} min required")
    
    # 1. Analizar solo ventana real
    print("\n1. Analizando ventana REAL...")
    real_results = analyze_window_optimized(
        signal_a, signal_b, fs, real_start_s, real_end_s, "Real",
        frequencies, BANDWIDTH_HZ, MAX_LAG_MS
    )
    
    # 2. Crear DataFrame con resultados
    print("\n2. Organizando resultados...")
    df_data = []
    for result in real_results:
        df_data.append({
            'Tipo': result['window'],
            'Frecuencia_Hz': result['frequency'],
            'Lag_ms': result['optimal_lag_ms'],
            'MRL_max': result['max_mrl']
        })
    
    df = pd.DataFrame(df_data)
    
    # 3. Guardar CSV
    csv_filename = f"mrl_optimizado_{window_name}.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"Resultados guardados en: {csv_filename}")
    
    # 4. Crear gráfico optimizado
    print("\n3. Creando gráfico...")
    plt.figure(figsize=(12, 7))
    
    plt.plot(
        df['Frecuencia_Hz'], 
        df['Lag_ms'],
        marker='o', 
        linewidth=2, 
        color='purple',
        label='Ventana Real'
    )
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Lag óptimo (ms)', fontsize=12)
    plt.title(f'Análisis MRL Optimizado - {window_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_filename = f"mrl_optimizado_{window_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_filename}")
    
    # 5. Resumen estadístico
    print("\n4. Resumen estadístico:")
    print(f"  Lag óptimo promedio: {df['Lag_ms'].mean():.2f} ± {df['Lag_ms'].std():.2f} ms")
    print(f"  MRL máximo promedio: {df['MRL_max'].mean():.4f} ± {df['MRL_max'].std():.4f}")
    print(f"  MRL máximo global: {df['MRL_max'].max():.4f} a {df.loc[df['MRL_max'].idxmax(), 'Frecuencia_Hz']} Hz")
    
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    return df

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == '__main__':
    print("Cargando datos...")
    start_total = time.time()
    
    # Cargar datos
    with h5py.File("allChan_phase_instantanea.mat", "r") as f:
        signal_data = np.array(f["allChan_phase_instantanea"])
    
    signal_data = signal_data.T  # (canales, muestras)
    
    print(f"Datos cargados: {signal_data.shape}")
    print(f"Duración total: {signal_data.shape[1]/FS/60:.1f} minutos")
    
    # Seleccionar canales
    signal_a = signal_data[0, :]   # Canal 1
    signal_b = signal_data[16, :]  # Canal 17
    
    print(f"Canales seleccionados: A shape={signal_a.shape}, B shape={signal_b.shape}")
    
    # Ejecutar análisis optimizado
    try:
        results = save_lagged_mrl_optimized(
            signal_a, signal_b, FS,
            "Real", 
            60,    # 1 minuto en segundos (prueba)
            360    # 6 minutos en segundos (prueba)
        )
        
        elapsed_total = time.time() - start_total
        print(f"\nTiempo total de ejecución: {elapsed_total/60:.1f} minutos")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()