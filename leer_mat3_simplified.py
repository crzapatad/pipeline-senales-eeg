import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# ==========================================
# CONFIGURACIÓN OPTIMIZADA
# ==========================================

# Parámetros optimizados para velocidad
FREQ_MIN = 1           # Frecuencia mínima (Hz)
FREQ_MAX = 10          # Frecuencia máxima (Hz) - reducido de 30 para velocidad
FREQ_STEP = 1          # Paso de frecuencia (Hz)
BANDWIDTH_HZ = 2       # Ancho de banda (Hz)
MAX_LAG_MS = 100       # Rango de lags (ms) - reducido de 250 para velocidad
FS = 500               # Frecuencia de muestreo (Hz)
WINDOW_MIN = 5         # Duración de ventana (minutos)

# ==========================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================

def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    """Filtro Butterworth pasabanda."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_lagged_mrl_by_window(signal_a, signal_b, fs, center_freqs=None, 
                                  window_width_hz=2.0, lags_ms=None, start_s=None, end_s=None):
    """Calcula el MRL entre dos señales para distintos desfases temporales y ventanas de frecuencia."""
    if center_freqs is None:
        center_freqs = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)
    if lags_ms is None:
        lags_ms = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1, 1)

    if start_s is None:
        start_s = 0
    if end_s is None:
        end_s = len(signal_a) / fs

    start_idx = int(start_s * fs)
    end_idx = int(end_s * fs)
    start_idx = max(0, min(start_idx, len(signal_a)))
    end_idx = max(start_idx + 1, min(end_idx, len(signal_a)))

    signal_a_seg = signal_a[start_idx:end_idx]
    signal_b_seg = signal_b[start_idx:end_idx]

    rows = []
    half_width = window_width_hz / 2.0

    for center_freq in center_freqs:
        lowcut = max(0.0, center_freq - half_width)
        highcut = min(30.0, center_freq + half_width)
        label = f"{int(center_freq)}Hz({int(lowcut)}-{int(highcut)}Hz)"

        filtered_a = butterworth_bandpass(signal_a_seg, fs, float(lowcut), float(highcut), order=4)
        filtered_b = butterworth_bandpass(signal_b_seg, fs, float(lowcut), float(highcut), order=4)

        for lag_ms in lags_ms:
            lag_samples = int(round(lag_ms / 1000.0 * fs))
            if lag_samples > 0:
                a_aligned = filtered_a[lag_samples:]
                b_aligned = filtered_b[:-lag_samples]
            elif lag_samples < 0:
                lag_abs = -lag_samples
                a_aligned = filtered_a[:-lag_abs]
                b_aligned = filtered_b[lag_abs:]
            else:
                a_aligned = filtered_a
                b_aligned = filtered_b

            if len(a_aligned) < 2 or len(b_aligned) < 2:
                continue

            analytic_a = hilbert(a_aligned)
            analytic_b = hilbert(b_aligned)
            phase_diff = np.unwrap(np.angle(analytic_b) - np.angle(analytic_a))
            mrl = np.abs(np.mean(np.exp(1j * phase_diff)))
            rows.append({"Ventana": label, "Frecuencia_Hz": float(center_freq), "Lag_ms": float(lag_ms), "MRL": float(mrl)})

    return pd.DataFrame(rows)

def save_lagged_mrl_analysis(signal_a, signal_b, fs, window_name, real_start_s, real_end_s,
                            center_freqs=None, window_width_hz=2.0, lags_ms=None):
    """
    Analiza una ventana REAL de 5 minutos para obtener
    el offset de MRL máximo por frecuencia.

    - Analiza la ventana de tiempo especificada
    - Calcula MRL para cada frecuencia y offset
    - Obtiene el offset donde el MRL es máximo
    - Sin comparación con ventanas aleatorias (versión optimizada)
    """

    ventana_s = 300.0  # 5 minutos

    # Configuración por defecto
    if center_freqs is None:
        center_freqs = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)

    if lags_ms is None:
        lags_ms = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1, 1)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    # Validaciones
    if abs((real_end_s - real_start_s) - ventana_s) > 1e-6:
        raise ValueError("La ventana REAL debe tener exactamente 5 minutos.")

    # Verificar duración suficiente
    total_duration = len(signal_a) / fs
    if total_duration < ventana_s:
        raise ValueError(
            f"El archivo tiene solo {total_duration/60:.1f} minutos, "
            f"pero se necesitan al menos {ventana_s/60:.0f} minutos."
        )

    print("\n" + "=" * 50)
    print("ANÁLISIS MRL - SOLO VENTANA REAL")
    print("=" * 50)
    print(f"Periodo REAL: {real_start_s/60:.2f} - {real_end_s/60:.2f} min")
    print(f"Frecuencias: {len(center_freqs)} ({center_freqs[0]}-{center_freqs[-1]} Hz)")
    print(f"Lags: {len(lags_ms)} ({lags_ms[0]} a {lags_ms[-1]} ms)")

    # Calcular ventana REAL
    real_start_idx = int(round(real_start_s * fs))
    real_end_idx = int(round(real_end_s * fs))

    real_a = signal_a[real_start_idx:real_end_idx]
    real_b = signal_b[real_start_idx:real_end_idx]

    print("Calculando MRL para ventana real...")

    real_df = compute_lagged_mrl_by_window(
        real_a, real_b, fs, center_freqs, window_width_hz, lags_ms, real_start_s, real_end_s
    )

    # Obtener offset máximo REAL por frecuencia
    real_maximos = real_df.loc[real_df.groupby("Ventana")["MRL"].idxmax()].copy()
    real_maximos["Tipo"] = "Real"
    real_maximos["Periodo"] = f"{real_start_s/60:.2f}-{real_end_s/60:.2f} min"

    # Guardar CSV
    archivo_csv = f"mrl_lagged_{window_name}_offset_maxima.csv"
    real_maximos.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
    print(f"Resultados guardados en: {archivo_csv}")

    # Crear gráfico simplificado
    plt.figure(figsize=(12, 7))

    # Agrupar por frecuencia
    curva = real_maximos.groupby("Frecuencia_Hz", as_index=False)["Lag_ms"].mean()

    plt.plot(curva["Frecuencia_Hz"], curva["Lag_ms"], marker="o", linewidth=2, color="purple", label="Ventana Real")

    # Configuración del gráfico
    plt.axhline(0, color="black", linestyle=":", alpha=0.6)
    plt.xlabel("Frecuencia (Hz)", fontsize=12)
    plt.ylabel("Offset máximo MRL (ms)", fontsize=12)
    plt.title(f"Análisis MRL - {window_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_filename = f"mrl_lagged_{window_name}_plot.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Gráfico guardado en: {plot_filename}")

    # Resumen estadístico
    print("\nResumen estadístico:")
    print(f"  Lag óptimo promedio: {real_maximos['Lag_ms'].mean():.2f} ± {real_maximos['Lag_ms'].std():.2f} ms")
    print(f"  MRL máximo promedio: {real_maximos['MRL'].mean():.4f} ± {real_maximos['MRL'].std():.4f}")
    print(f"  MRL máximo global: {real_maximos['MRL'].max():.4f}")

    print("\n" + "=" * 50)
    print("ANÁLISIS COMPLETADO")
    print("=" * 50)

    return real_maximos

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == '__main__':
    print("Cargando datos...")
    
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
        print("\nParámetros optimizados:")
        print(f"  Rango frecuencias: {FREQ_MIN}-{FREQ_MAX} Hz")
        print(f"  Rango de lags: ±{MAX_LAG_MS} ms")
        print(f"  Sin comparación de ventanas aleatorias")
        
        results = save_lagged_mrl_analysis(
            signal_a, signal_b, FS,
            "Real", 
            60,    # 1 minuto en segundos (prueba)
            360    # 6 minutos en segundos (prueba)
        )
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()