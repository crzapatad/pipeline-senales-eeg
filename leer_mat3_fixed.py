import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import time

# ==========================================
# CONFIGURACIÓN OPTIMIZADA
# ==========================================

# Parámetros optimizados para velocidad
FREQ_MIN = 1           # Frecuencia mínima (Hz)
FREQ_MAX = 10          # Frecuencia máxima (Hz) - reducido de 30
FREQ_STEP = 1          # Paso de frecuencia (Hz)
BANDWIDTH_HZ = 2       # Ancho de banda (Hz)
MAX_LAG_MS = 100       # Rango de lags (ms) - reducido de 250
FS = 500               # Frecuencia de muestreo (Hz)
WINDOW_MIN = 5         # Duración de ventana (minutos)
N_RANDOM_WINDOWS = 1   # Número de ventanas aleatorias - reducido de 3

# ==========================================
# FUNCIONES OPTIMIZADAS
# ==========================================

def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    """Filtro Butterworth pasabanda."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_lagged_mrl_by_window_optimized(signal_a, signal_b, fs, center_freqs=None, 
                                          window_width_hz=2.0, lags_ms=None, start_s=None, end_s=None):
    """Versión optimizada de compute_lagged_mrl_by_window."""
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
            # CORRECCIÓN: Agregar columna Frecuencia_Hz
            rows.append({"Ventana": label, "Frecuencia_Hz": float(center_freq), "Lag_ms": float(lag_ms), "MRL": float(mrl)})

    return pd.DataFrame(rows)

def save_lagged_mrl_5min_comparison_fixed(signal_a, signal_b, fs, window_name, real_start_s, real_end_s,
                                         pool_start_s=0.0, pool_end_s=None, center_freqs=None,
                                         window_width_hz=2.0, lags_ms=None, random_seed=20260817,
                                         n_random_windows=1):
    """Versión corregida y optimizada del análisis MRL."""
    
    ventana_s = 300.0  # 5 minutos

    if center_freqs is None:
        center_freqs = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)

    if lags_ms is None:
        lags_ms = np.arange(-MAX_LAG_MS, MAX_LAG_MS + 1, 1)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    if pool_end_s is None:
        pool_end_s = len(signal_a) / fs

    # Validaciones
    if abs((real_end_s - real_start_s) - ventana_s) > 1e-6:
        raise ValueError("La ventana REAL debe tener exactamente 5 minutos.")

    if pool_end_s - pool_start_s < ventana_s:
        raise ValueError("El periodo disponible es menor que 5 minutos.")

    if real_start_s < pool_start_s or real_end_s > pool_end_s:
        raise ValueError("La ventana real debe estar dentro del periodo disponible.")

    # Función para comprobar solapamiento
    def se_solapan(inicio1, fin1, inicio2, fin2):
        return max(inicio1, inicio2) < min(fin1, fin2)

    # Generador aleatorio
    rng = np.random.default_rng(random_seed)
    max_start = pool_end_s - ventana_s

    # Buscar ventanas aleatorias NO solapadas
    random_windows = []
    intentos = 0
    max_intentos = 10000

    while len(random_windows) < n_random_windows and intentos < max_intentos:
        intentos += 1

        candidato = float(rng.uniform(pool_start_s, max_start))
        candidato_fin = candidato + ventana_s

        # No puede coincidir con la ventana real
        if se_solapan(candidato, candidato_fin, real_start_s, real_end_s):
            continue

        # No puede coincidir con ninguna ventana aleatoria anterior
        conflicto = False
        for inicio_existente, fin_existente in random_windows:
            if se_solapan(candidato, candidato_fin, inicio_existente, fin_existente):
                conflicto = True
                break

        if not conflicto:
            random_windows.append((candidato, candidato_fin))

    if len(random_windows) < n_random_windows:
        print(f"ADVERTENCIA: Solo se generaron {len(random_windows)} de {n_random_windows} ventanas aleatorias")

    # Ordenar cronológicamente
    random_windows.sort()

    print("\n==============================================")
    print("COMPARACIÓN REAL VS VENTANAS ALEATORIAS")
    print("==============================================")
    print(f"Periodo REAL: {real_start_s/60:.2f} - {real_end_s/60:.2f} min")

    for i, (inicio, fin) in enumerate(random_windows, 1):
        print(f"Azar {i}: {inicio/60:.2f} - {fin/60:.2f} min")

    # Calcular REAL
    print("\nCalculando ventana REAL...")
    real_start_idx = int(round(real_start_s * fs))
    real_end_idx = int(round(real_end_s * fs))

    real_a = signal_a[real_start_idx:real_end_idx]
    real_b = signal_b[real_start_idx:real_end_idx]

    real_df = compute_lagged_mrl_by_window_optimized(
        real_a, real_b, fs, center_freqs, window_width_hz, lags_ms
    )

    # Obtener offset máximo REAL por frecuencia
    real_maximos = real_df.loc[real_df.groupby("Ventana")["MRL"].idxmax()].copy()
    real_maximos["Tipo"] = "Real"
    real_maximos["Periodo"] = f"{real_start_s/60:.2f}-{real_end_s/60:.2f} min"

    # Calcular las ventanas aleatorias
    random_results = []
    for i, (inicio_s, fin_s) in enumerate(random_windows, start=1):
        print(f"Calculando ventana ALEATORIA {i}...")
        inicio_idx = int(round(inicio_s * fs))
        fin_idx = int(round(fin_s * fs))

        random_a = signal_a[inicio_idx:fin_idx]
        random_b = signal_b[inicio_idx:fin_idx]

        df_random = compute_lagged_mrl_by_window_optimized(
            random_a, random_b, fs, center_freqs, window_width_hz, lags_ms
        )

        random_maximos = df_random.loc[df_random.groupby("Ventana")["MRL"].idxmax()].copy()
        random_maximos["Tipo"] = f"Azar {i}"
        random_maximos["Periodo"] = f"{inicio_s/60:.2f}-{fin_s/60:.2f} min"
        random_results.append(random_maximos)

    # Combinar todos los resultados
    todos = pd.concat([real_maximos] + random_results, ignore_index=True)

    # Guardar CSV
    archivo_csv = f"mrl_lagged_{window_name}_real_vs_{n_random_windows}_azar_offset_maxima.csv"
    todos.to_csv(archivo_csv, index=False, encoding="utf-8-sig")
    print(f"\nResultados guardados en: {archivo_csv}")

    # Crear gráfico
    plt.figure(figsize=(12, 7))
    tipos = ["Real"] + [f"Azar {i+1}" for i in range(len(random_windows))]

    for tipo in tipos:
        datos_tipo = todos[todos["Tipo"] == tipo]
        # CORRECCIÓN: Usar columna Frecuencia_Hz que ahora existe
        curva = datos_tipo.groupby("Frecuencia_Hz", as_index=False)["Lag_ms"].mean()
        plt.plot(curva["Frecuencia_Hz"], curva["Lag_ms"], marker="o", linewidth=2, label=tipo)

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

    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    return todos

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
    
    # Ejecutar análisis corregido
    try:
        print("\nParámetros optimizados:")
        print(f"  Rango frecuencias: {FREQ_MIN}-{FREQ_MAX} Hz")
        print(f"  Rango de lags: ±{MAX_LAG_MS} ms")
        print(f"  Ventanas aleatorias: {N_RANDOM_WINDOWS}")
        
        results = save_lagged_mrl_5min_comparison_fixed(
            signal_a, signal_b, FS,
            "Real", 
            60,    # 1 minuto en segundos (prueba)
            360,   # 6 minutos en segundos (prueba)
            n_random_windows=N_RANDOM_WINDOWS
        )
        
        elapsed_total = time.time() - start_total
        print(f"\nTiempo total de ejecución: {elapsed_total/60:.1f} minutos")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()