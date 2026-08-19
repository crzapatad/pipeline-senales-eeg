import glob
import math
import os
import re
import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch, spectrogram, hilbert, butter, filtfilt, resample_poly, correlate, sosfiltfilt
from scipy.stats import t
from pathlib import Path
from joblib import Parallel, delayed

# ======================================
# Clasificación de bandas EEG
# ======================================

def clasificar_banda(f):
    if 0.5 <= f < 4:
        return "Delta"
    elif 4 <= f < 8:
        return "Theta"
    elif 8 <= f < 13:
        return "Alpha"
    elif 13 <= f < 30:
        return "Beta"
    elif 30 <= f < 50:
        return "Gamma baja"
    elif 50 <= f <= 100:
        return "Gamma alta"
    else:
        return "Fuera de banda"


def parse_open_ephys_header(header_text):
    header = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("header."):
            continue
        parts = line.split("=", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().replace("header.", "")
        value = parts[1].strip().rstrip(";")
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        header[key] = value
    return header


def read_open_ephys_continuous(path):
    with open(path, "rb") as f:
        header_bytes = f.read(1024)
        header_text = header_bytes.decode("latin1", errors="replace")
        header = parse_open_ephys_header(header_text)
        sample_rate = float(header.get("sampleRate", 0))
        bit_volts = float(header.get("bitVolts", 1.0))

        data_segments = []
        while True:
            record_header = f.read(12)
            if not record_header or len(record_header) < 12:
                break
            timestamp = int.from_bytes(record_header[0:8], "little", signed=True)
            num_samples = int.from_bytes(record_header[8:10], "little", signed=False)
            recording_number = int.from_bytes(record_header[10:12], "little", signed=False)
            samples_bytes = f.read(2 * num_samples)
            if len(samples_bytes) < 2 * num_samples:
                break
            samples = np.frombuffer(samples_bytes, dtype="<i2").astype(np.float64) * bit_volts
            marker = f.read(10)
            data_segments.append(samples)

        if len(data_segments) == 0:
            raise ValueError(f"No se pudieron leer muestras de {path}")
        data = np.concatenate(data_segments)
        return sample_rate, data, header


def subsample_open_ephys(data, original_fs, target_fs):
    if target_fs <= 0 or target_fs >= original_fs:
        return data
    gcd = math.gcd(int(original_fs), int(target_fs))
    up = target_fs // gcd
    down = original_fs // gcd
    if up == 0 or down == 0:
        raise ValueError(f"Tasas de muestreo inválidas: original={original_fs}, target={target_fs}")
    return resample_poly(data, up, down)


def filter_and_downsample_matrix(matriz_eeg, original_fs, desired_fs, order=2, f1=0.1, f2=150.0):
    """Filtra en banda y reduce la tasa de muestreo de una matriz EEG (n_samples x n_channels).

    Filtra cada canal con un filtro Butterworth pasabanda (orden `order`, cortes f1-f2)
    y luego hace downsampling por promedio de ventanas (factor = original_fs/desired_fs).
    Devuelve la matriz reducida y la nueva frecuencia de muestreo.
    """
    if desired_fs is None or desired_fs >= original_fs:
        return matriz_eeg, original_fs

    nyq = 0.5 * float(original_fs)
    Wn = np.array([f1 / nyq, f2 / nyq])
    b, a = butter(order, Wn, btype="bandpass")

    n_samples, n_channels = matriz_eeg.shape
    # filtrar cada canal
    filtered = np.zeros_like(matriz_eeg)
    for ch in range(n_channels):
        # filtfilt en cada canal; usar padlen=None tal como en el fragmento original
        filtered[:, ch] = filtfilt(b, a, matriz_eeg[:, ch], padlen=None)

    downsample_factor = int(original_fs // desired_fs)
    if downsample_factor <= 1:
        return filtered, original_fs

    multiplo = n_samples // downsample_factor
    if multiplo == 0:
        raise ValueError("La señal es demasiado corta para el factor de downsampling solicitado")

    trimmed = filtered[:downsample_factor * multiplo, :]
    # reshape a (multiplo, downsample_factor, n_channels) y promediar sobre axis=1
    reshaped = trimmed.reshape(multiplo, downsample_factor, n_channels)
    down = reshaped.mean(axis=1)

    return down, desired_fs


def load_lfp_channels(lfp_dir, target_fs=None):
    continuous_files = sorted(
        [p for p in glob.glob(os.path.join(lfp_dir, "*.continuous")) if not os.path.basename(p).startswith("._")],
        key=lambda x: int(re.search(r"CH(\d+)", os.path.basename(x), re.IGNORECASE).group(1))
    )
    if len(continuous_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .continuous en {lfp_dir}")

    channel_data = []
    sample_rate = None
    channel_names = []
    for path in continuous_files:
        fs, data, header = read_open_ephys_continuous(path)
        if sample_rate is None:
            sample_rate = fs
        elif sample_rate != fs:
            raise ValueError(f"Frecuencias de muestreo inconsistentes en {path}")
        if target_fs is not None and target_fs < sample_rate:
            data = subsample_open_ephys(data, int(sample_rate), int(target_fs))
        channel_data.append(data)
        channel_names.append(header.get("channel", os.path.basename(path)))

    min_len = min(len(ch) for ch in channel_data)
    if any(len(ch) != min_len for ch in channel_data):
        channel_data = [ch[:min_len] for ch in channel_data]

    matriz_eeg = np.vstack(channel_data).T
    return matriz_eeg, sample_rate if target_fs is None else target_fs, channel_names


def load_mat_data(file_path):
    with h5py.File(file_path, "r") as datos:
        config = None
        if "processing" in datos and "channel_configuration" in datos["processing"]:
            config = {region: np.array(datos["processing"]["channel_configuration"][region]) for region in datos["processing"]["channel_configuration"].keys()}
        matriz_eeg = np.array(datos["allChan_clean"])
        Fs = float(np.array(datos["Fs"])[0][0])
        return matriz_eeg, Fs, config


def detect_data_source(target_fs=None, search_parents=True):
    """Busca un archivo .mat o una carpeta LFP en el directorio actual y (opcionalmente) en los padres.

    Retorna la primera coincidencia encontrada.
    """
    start_dir = Path(os.getcwd())
    search_dirs = [start_dir]
    if search_parents:
        search_dirs += list(start_dir.parents)

    for d in search_dirs:
        # buscar .mat
        mat_files = [f for f in glob.glob(os.path.join(str(d), "*.mat")) if not os.path.basename(f).startswith("._")]
        if len(mat_files) > 0:
            # intentar abrir cada .mat hasta encontrar uno válido con la estructura esperada
            for mf in mat_files:
                try:
                    data = load_mat_data(mf)
                except Exception as e:
                    print(f"Archivo .mat ignorado (no válido para carga): {mf} -> {e}")
                    continue
                print("Se detectó archivo .mat válido. Usando:")
                print(mf)
                return data

        # buscar carpeta LFP
        lfp_dir = d / "LFP"
        if lfp_dir.is_dir():
            print(f"Leyendo archivos Open Ephys en {lfp_dir}")
            matriz_eeg, Fs, channel_names = load_lfp_channels(str(lfp_dir), target_fs=target_fs)
            return matriz_eeg, Fs, None

    raise FileNotFoundError("No se encontró un archivo .mat ni la carpeta LFP con archivos .continuous en el árbol de directorios.")


def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    if low <= 1e-6:
        sos = butter(order, high, btype="low", output="sos")
    else:
        sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def compute_phase_difference_by_windows(signal_a, signal_b, fs, center_freqs=None, window_width_hz=2.0):
    """Calcula la diferencia de fase instantánea para ventanas centradas en frecuencias 1..29 Hz.

    Cada ventana usa un pasabanda de ancho 2 Hz alrededor de la frecuencia central.
    Ejemplos: 1 Hz -> 0-2 Hz, 2 Hz -> 1-3 Hz, ..., 29 Hz -> 28-30 Hz.

    Devuelve:
    - window_labels: etiquetas de las ventanas, por ejemplo 1Hz(0-2Hz), 2Hz(1-3Hz), ...
    - phase_diff_matrix: matriz de forma (n_windows, n_samples)
    """
    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)
    center_freqs = np.asarray(center_freqs, dtype=float)

    n_samples = len(signal_a)
    phase_diff_matrix = np.zeros((len(center_freqs), n_samples), dtype=float)
    window_labels = []

    half_width = window_width_hz / 2.0
    for idx, center_freq in enumerate(center_freqs):
        lowcut = max(0.0, center_freq - half_width)
        highcut = min(30.0, center_freq + half_width)
        label = f"{int(center_freq)}Hz({int(lowcut)}-{int(highcut)}Hz)"
        window_labels.append(label)

        filtered_a = butterworth_bandpass(signal_a, fs, float(lowcut), float(highcut), order=4)
        filtered_b = butterworth_bandpass(signal_b, fs, float(lowcut), float(highcut), order=4)

        analytic_a = hilbert(filtered_a)
        analytic_b = hilbert(filtered_b)

        phase_diff = np.unwrap(np.angle(analytic_b) - np.angle(analytic_a))
        phase_diff_matrix[idx, :] = phase_diff

    return window_labels, phase_diff_matrix

# ==============================================================================
# 1. FUNCIÓN PARA PROCESAR UN SOLO PAR DE CANALES (PARALELIZABLE)
# ==============================================================================
def calcular_mrl_par_individual(par, fases_matriz, lags):
    """
    Calcula el Mean Resultant Length (MRL) rezagado para un único par de canales.
    
    Parámetros:
    -----------
    par : tuple (int, int)
        Índices de los dos canales a comparar (ej: (0, 1)).
    fases_matriz : np.ndarray
        Matriz con las fases continuas de los canales en formato (n_canales, n_muestras).
    lags : array-like
        Arreglo de rezagos en número de muestras (ej: range(-50, 51)).
        
    Retorna:
    --------
    tuple : (par, mrl_lags)
        Identificador del par y un arreglo 1D con el MRL calculado para cada lag.
    """
    ch1_idx, ch2_idx = par
    fase1 = fases_matriz[ch1_idx]
    fase2 = fases_matriz[ch2_idx]
    n_samples = len(fase1)
    
    mrl_lags = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Alineación de señales según el rezago (lag)
        if lag < 0:
            s1 = fase1[-lag:]
            s2 = fase2[:n_samples + lag]
        elif lag > 0:
            s1 = fase1[:n_samples - lag]
            s2 = fase2[lag:]
        else:
            s1 = fase1
            s2 = fase2
            
        # Diferencia de fase compleja e instantánea: exp(i * (phase1 - phase2))
        diferencia_fase = np.exp(1j * (s1 - s2))
        
        # MRL es la magnitud del vector promedio resultante
        mrl_lags[i] = np.abs(np.mean(diferencia_fase))
        
    return par, mrl_lags


# ==============================================================================
# 2. FUNCIÓN AUXILIAR: VERIFICAR SOLAPAMIENTO DE TIEMPOS
# ==============================================================================
def _se_solapan(inicio1, fin1, inicio2, fin2):
    """Devuelve True si dos rangos de tiempo se cruzan."""
    return max(inicio1, inicio2) < min(fin1, fin2)


# ==============================================================================
# 3. FUNCIÓN PRINCIPAL DE PROCESAMIENTO Y PARALELIZACIÓN
# ==============================================================================
def save_lagged_mrl_5min_three_random_comparison(
    filepath, 
    output_filepath="resultado_mrl.mat",
    fs=500, 
    real_window_min=(11, 16), 
    lag_max_sec=0.1, 
    num_random_windows=1,  # <-- CAMBIADO A 1 PARA QUE NO FALLE SI EL ARCHIVO ES CORTO
    n_jobs=-1
):
    print(f"Cargando archivo: {filepath}...")
    data_mat = sio.loadmat(filepath)
    
    keys = [k for k in data_mat.keys() if not k.startswith('__')]
    eeg_data = data_mat[keys[0]]
    
    n_canales, total_muestras = eeg_data.shape
    duracion_total_sec = total_muestras / fs
    print(f"EEG cargado: {n_canales} canales, {total_muestras} muestras ({duracion_total_sec/60:.2f} min).")
    
    print("Calculando transformada de Hilbert para extraer la fase...")
    fases_eeg = np.angle(hilbert(eeg_data, axis=1))
    
    win_sec = 300 
    real_start_sec = real_window_min[0] * 60
    real_end_sec = real_window_min[1] * 60
    
    if real_end_sec > duracion_total_sec:
        raise ValueError(f"Error: La ventana real termina en el min {real_window_min[1]}, pero el archivo solo dura {duracion_total_sec/60:.2f} min.")
    
    random_windows = []
    intentos = 0
    
    np.random.seed(42)
    print(f"Buscando {num_random_windows} ventana(s) aleatoria(s) de 5 min...")
    
    while len(random_windows) < num_random_windows:
        intentos += 1
        if intentos > 10000:
            raise RuntimeError(f"El archivo es muy corto para extraer {num_random_windows} ventanas aleatorias extra de 5 min.")
        
        candidato_inicio = np.random.uniform(0, duracion_total_sec - win_sec)
        candidato_fin = candidato_inicio + win_sec
        
        if _se_solapan(candidato_inicio, candidato_fin, real_start_sec, real_end_sec): continue
            
        conflicto = False
        for r_ini, r_fin in random_windows:
            if _se_solapan(candidato_inicio, candidato_fin, r_ini, r_fin):
                conflicto = True
                break
        if conflicto: continue
            
        random_windows.append((candidato_inicio, candidato_fin))

    ventanas_totales = {'Real': (real_start_sec, real_end_sec)}
    for idx, (r_ini, r_fin) in enumerate(random_windows, 1):
        ventanas_totales[f'Aleatoria_{idx}'] = (r_ini, r_fin)

    pares = list(combinations(range(n_canales), 2))
    lags = np.arange(-int(lag_max_sec * fs), int(lag_max_sec * fs) + 1)
    
    resultados_finales = {}
    for nombre_ventana, (t_start, t_end) in ventanas_totales.items():
        print(f"\n---> Calculando MRL ventana [{nombre_ventana}] en PARALELO...")
        fases_segmento = fases_eeg[:, int(t_start * fs):int(t_end * fs)]
        mrl_paralelo = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(calcular_mrl_par_individual)(par, fases_segmento, lags) for par in pares
        )
        
        matriz_mrl = np.zeros((len(pares), len(lags)))
        for idx, (par, mrl_vector) in enumerate(mrl_paralelo):
            matriz_mrl[idx, :] = mrl_vector
        resultados_finales[nombre_ventana] = matriz_mrl

    resultados_finales['lags_sec'] = lags / fs
    resultados_finales['pares'] = np.array(pares)
    sio.savemat(output_filepath, resultados_finales)
    print(f"\n¡Completado! Guardado en: {output_filepath}")


def compute_lagged_mrl_by_window(signal_a, signal_b, fs, center_freqs=None, window_width_hz=2.0,
                                lags_ms=None, start_s=None, end_s=None):
    """Calcula el MRL entre dos señales para distintos desfases temporales y ventanas de frecuencia."""
    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)
    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1)

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
            rows.append({"Ventana": label, "Lag_ms": float(lag_ms), "MRL": float(mrl)})

    return pd.DataFrame(rows)


def plot_lagged_mrl_summaries(df_lagged, window_name):
    """Genera el heatmap ordenado por frecuencia, curvas MRL por desfase, y máximos por frecuencia/tiempo."""
    matrix_lagged = df_lagged.pivot(index="Ventana", columns="Lag_ms", values="MRL")

    freq_values = []
    for label in matrix_lagged.index:
        match = re.search(r"(\d+)Hz\(", label)
        if match is None:
            raise ValueError(f"No se pudo extraer la frecuencia desde la etiqueta: {label}")
        freq_values.append(int(match.group(1)))
    freq_values = np.asarray(freq_values, dtype=float)

    order = np.argsort(freq_values)
    freq_values_sorted = freq_values[order]
    matrix_lagged_sorted = matrix_lagged.iloc[order, :].copy()

    lag_values = np.asarray(matrix_lagged_sorted.columns, dtype=float)
    col_order = np.argsort(lag_values)
    lag_values = lag_values[col_order]
    matrix_lagged_sorted = matrix_lagged_sorted.iloc[:, col_order].copy()
    matrix_lagged_sorted.columns = lag_values

    matrix_lagged_sorted.to_csv(f"mrl_lagged_{window_name}_matrix.csv", encoding="utf-8-sig")
    np.save(f"mrl_lagged_{window_name}_matrix.npy", matrix_lagged_sorted.values)

    plt.figure(figsize=(14, 8))
    plt.imshow(matrix_lagged_sorted.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="MRL")
    tick_step = max(1, len(lag_values) // 10)
    plt.xticks(np.arange(0, len(lag_values), tick_step), [f"{x:.0f}" for x in lag_values[::tick_step]])
    plt.yticks(np.arange(len(freq_values_sorted)), [f"{int(x)}Hz" for x in freq_values_sorted])
    plt.xlabel("Desfase (ms)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title(f"MRL por desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap("viridis")
    for idx, lag_value in enumerate(lag_values):
        color = cmap(idx / max(1, len(lag_values) - 1))
        plt.plot(freq_values_sorted, matrix_lagged_sorted.iloc[:, idx].to_numpy(), color=color, alpha=0.25, linewidth=1.0)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("MRL")
    plt.title(f"MRL por frecuencia para cada desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_curves_by_lag.png", dpi=300, bbox_inches="tight")
    plt.close()

    row_max_idx = np.argmax(matrix_lagged_sorted.values, axis=1)
    row_max_vals = matrix_lagged_sorted.values[np.arange(len(freq_values_sorted)), row_max_idx]
    row_max_lags = lag_values[row_max_idx]
    df_freq_max = pd.DataFrame({
        "Frecuencia_Hz": freq_values_sorted,
        "Lag_ms_Max_MRL": row_max_lags,
        "MRL_Maximo": row_max_vals,
    })
    df_freq_max.to_csv(f"mrl_lagged_{window_name}_freq_maxima.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(freq_values_sorted, row_max_lags, marker="o", linewidth=1.5, color="tab:blue")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Desfase (ms) donde el MRL es máximo")
    plt.title(f"Desfase de máximo MRL por frecuencia - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_freq_maxima.png", dpi=300, bbox_inches="tight")
    plt.close()

    col_max_idx = np.argmax(matrix_lagged_sorted.values, axis=0)
    col_max_vals = matrix_lagged_sorted.values[col_max_idx, np.arange(len(lag_values))]
    col_max_freqs = freq_values_sorted[col_max_idx]
    df_time_max = pd.DataFrame({
        "Lag_ms": lag_values,
        "Frecuencia_Hz_Max_MRL": col_max_freqs,
        "MRL_Maximo": col_max_vals,
    })
    df_time_max.to_csv(f"mrl_lagged_{window_name}_time_maxima.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(lag_values, col_max_freqs, marker="o", linewidth=1.5, color="tab:orange")
    plt.xlabel("Desfase (ms)")
    plt.ylabel("Frecuencia (Hz) donde el MRL es máximo")
    plt.title(f"Frecuencia de máximo MRL por desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_time_maxima.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Análisis MRL guardado para {window_name}: {f'mrl_lagged_{window_name}_heatmap.png'}, {f'mrl_lagged_{window_name}_curves_by_lag.png'}, {f'mrl_lagged_{window_name}_freq_maxima.png'}, {f'mrl_lagged_{window_name}_time_maxima.png'}")


def compute_lagged_envelope_correlation_by_window(signal_a, signal_b, fs, center_freqs=None, window_width_hz=2.0,
                                                    lags_ms=None, start_s=None, end_s=None):
    """Calcula la correlación entre las envolventes de amplitud de dos señales para distintos
    desfases temporales y ventanas de frecuencia.

    Para cada frecuencia central en `center_freqs` se filtra cada señal con un pasabanda de ancho
    `window_width_hz` (por ejemplo 2 Hz: 1 Hz -> 0-2 Hz, 2 Hz -> 1-3 Hz, ...) y se obtiene la
    envolvente de amplitud mediante la transformada de Hilbert (módulo de la señal analítica).
    Luego, para cada desfase en `lags_ms` (por ejemplo -250 a +250 ms en pasos de 1 ms), se
    desplazan las envolventes entre sí y se calcula el coeficiente de correlación de Pearson.

    Devuelve un DataFrame con columnas: Ventana, Lag_ms, Correlacion_Envolvente.
    """
    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)
    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1, dtype=float)

    if start_s is None:
        start_s = 0
    if end_s is None:
        end_s = len(signal_a) / fs

    start_idx = int(round(start_s * fs))
    end_idx = int(round(end_s * fs))
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

        envolvente_a = np.abs(hilbert(filtered_a))
        envolvente_b = np.abs(hilbert(filtered_b))

        for lag_ms in lags_ms:
            lag_samples = int(round(lag_ms / 1000.0 * fs))
            if lag_samples > 0:
                a_aligned = envolvente_a[lag_samples:]
                b_aligned = envolvente_b[:-lag_samples]
            elif lag_samples < 0:
                lag_abs = -lag_samples
                a_aligned = envolvente_a[:-lag_abs]
                b_aligned = envolvente_b[lag_abs:]
            else:
                a_aligned = envolvente_a
                b_aligned = envolvente_b

            if len(a_aligned) < 2 or len(b_aligned) < 2:
                continue

            if np.std(a_aligned) == 0 or np.std(b_aligned) == 0:
                correlacion = np.nan
            else:
                correlacion = float(np.corrcoef(a_aligned, b_aligned)[0, 1])

            rows.append({"Ventana": label, "Lag_ms": float(lag_ms), "Correlacion_Envolvente": correlacion})

    return pd.DataFrame(rows)


def plot_lagged_envelope_summaries(df_lagged, window_name):
    """Genera el heatmap ordenado por frecuencia, curvas de correlación de envolvente por desfase,
    y los máximos por frecuencia/tiempo, análogo a plot_lagged_mrl_summaries pero para envolvente."""
    matrix_lagged = df_lagged.pivot(index="Ventana", columns="Lag_ms", values="Correlacion_Envolvente")

    freq_values = []
    for label in matrix_lagged.index:
        match = re.search(r"(\d+)Hz\(", label)
        if match is None:
            raise ValueError(f"No se pudo extraer la frecuencia desde la etiqueta: {label}")
        freq_values.append(int(match.group(1)))
    freq_values = np.asarray(freq_values, dtype=float)

    order = np.argsort(freq_values)
    freq_values_sorted = freq_values[order]
    matrix_lagged_sorted = matrix_lagged.iloc[order, :].copy()

    lag_values = np.asarray(matrix_lagged_sorted.columns, dtype=float)
    col_order = np.argsort(lag_values)
    lag_values = lag_values[col_order]
    matrix_lagged_sorted = matrix_lagged_sorted.iloc[:, col_order].copy()
    matrix_lagged_sorted.columns = lag_values

    matrix_lagged_sorted.to_csv(f"envelope_lagged_{window_name}_matrix.csv", encoding="utf-8-sig")
    np.save(f"envelope_lagged_{window_name}_matrix.npy", matrix_lagged_sorted.values)

    plt.figure(figsize=(14, 8))
    plt.imshow(matrix_lagged_sorted.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Correlación de envolvente (r)")
    tick_step = max(1, len(lag_values) // 10)
    plt.xticks(np.arange(0, len(lag_values), tick_step), [f"{x:.0f}" for x in lag_values[::tick_step]])
    plt.yticks(np.arange(len(freq_values_sorted)), [f"{int(x)}Hz" for x in freq_values_sorted])
    plt.xlabel("Desfase (ms)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title(f"Correlación de envolvente por desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"envelope_lagged_{window_name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap("viridis")
    for idx, lag_value in enumerate(lag_values):
        color = cmap(idx / max(1, len(lag_values) - 1))
        plt.plot(freq_values_sorted, matrix_lagged_sorted.iloc[:, idx].to_numpy(), color=color, alpha=0.25, linewidth=1.0)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Correlación de envolvente (r)")
    plt.title(f"Correlación de envolvente por frecuencia para cada desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"envelope_lagged_{window_name}_curves_by_lag.png", dpi=300, bbox_inches="tight")
    plt.close()

    valores = matrix_lagged_sorted.values
    row_max_idx = np.nanargmax(valores, axis=1)
    row_max_vals = valores[np.arange(len(freq_values_sorted)), row_max_idx]
    row_max_lags = lag_values[row_max_idx]
    df_freq_max = pd.DataFrame({
        "Frecuencia_Hz": freq_values_sorted,
        "Lag_ms_Max_Correlacion": row_max_lags,
        "Correlacion_Maxima": row_max_vals,
    })
    df_freq_max.to_csv(f"envelope_lagged_{window_name}_freq_maxima.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(freq_values_sorted, row_max_lags, marker="o", linewidth=1.5, color="tab:blue")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Desfase (ms) donde la correlación de envolvente es máxima")
    plt.title(f"Desfase de máxima correlación de envolvente por frecuencia - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"envelope_lagged_{window_name}_freq_maxima.png", dpi=300, bbox_inches="tight")
    plt.close()

    col_max_idx = np.nanargmax(valores, axis=0)
    col_max_vals = valores[col_max_idx, np.arange(len(lag_values))]
    col_max_freqs = freq_values_sorted[col_max_idx]
    df_time_max = pd.DataFrame({
        "Lag_ms": lag_values,
        "Frecuencia_Hz_Max_Correlacion": col_max_freqs,
        "Correlacion_Maxima": col_max_vals,
    })
    df_time_max.to_csv(f"envelope_lagged_{window_name}_time_maxima.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(lag_values, col_max_freqs, marker="o", linewidth=1.5, color="tab:orange")
    plt.xlabel("Desfase (ms)")
    plt.ylabel("Frecuencia (Hz) donde la correlación de envolvente es máxima")
    plt.title(f"Frecuencia de máxima correlación de envolvente por desfase temporal - ventana {window_name}")
    plt.tight_layout()
    plt.savefig(f"envelope_lagged_{window_name}_time_maxima.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Análisis de correlación de envolvente guardado para {window_name}: "
        f"envelope_lagged_{window_name}_heatmap.png, envelope_lagged_{window_name}_curves_by_lag.png, "
        f"envelope_lagged_{window_name}_freq_maxima.png, envelope_lagged_{window_name}_time_maxima.png"
    )


def compute_lagged_mrl_curve_from_phase(phase_a, phase_b, fs, lags_ms):
    """Calcula MRL para todos los desfases usando correlacion compleja."""
    phase_a = np.asarray(phase_a, dtype=float)
    phase_b = np.asarray(phase_b, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    if len(phase_a) != len(phase_b):
        raise ValueError("Las fases deben tener la misma longitud")

    n_samples = len(phase_a)
    lag_samples = np.rint(lags_ms / 1000.0 * fs).astype(int)

    unit_a = np.exp(1j * phase_a)
    unit_b = np.exp(1j * phase_b)
    corr_full = correlate(unit_b, unit_a, mode="full", method="fft")
    center_idx = n_samples - 1

    mrl_values = np.full(len(lags_ms), np.nan, dtype=float)
    valid = np.abs(lag_samples) < n_samples
    corr_idx = center_idx - lag_samples[valid]
    overlap = n_samples - np.abs(lag_samples[valid])
    mrl_values[valid] = np.abs(corr_full[corr_idx]) / overlap

    return mrl_values


def summarize_offset_maxima(offset_maxima):
    """Promedia offsets por frecuencia y calcula IC 95% con t de Student."""
    summary_rows = []
    grouped = offset_maxima.groupby(["Frecuencia_Hz", "Ventana_Frecuencia"], sort=True)

    for (freq_hz, freq_label), group in grouped:
        offsets = group["Offset_ms_Max_MRL"].dropna().to_numpy(dtype=float)
        max_mrl = group["MRL_Maximo"].dropna().to_numpy(dtype=float)
        n = len(offsets)

        if n == 0:
            mean_offset = np.nan
            std_offset = np.nan
            sem_offset = np.nan
            ci_half_width = np.nan
        elif n == 1:
            mean_offset = float(np.mean(offsets))
            std_offset = 0.0
            sem_offset = 0.0
            ci_half_width = 0.0
        else:
            mean_offset = float(np.mean(offsets))
            std_offset = float(np.std(offsets, ddof=1))
            sem_offset = std_offset / math.sqrt(n)
            ci_half_width = float(t.ppf(0.975, n - 1) * sem_offset)

        summary_rows.append({
            "Frecuencia_Hz": float(freq_hz),
            "Ventana_Frecuencia": freq_label,
            "N_Ventanas": int(n),
            "Duracion_Ventana_s": float(group["Duracion_Ventana_s"].iloc[0]),
            "Salto_Ventana_s": float(group["Salto_Ventana_s"].iloc[0]),
            "Offset_ms_Promedio": mean_offset,
            "Offset_ms_SD": std_offset,
            "Offset_ms_SEM": sem_offset,
            "Offset_ms_IC95_Lower": mean_offset - ci_half_width if n > 0 else np.nan,
            "Offset_ms_IC95_Upper": mean_offset + ci_half_width if n > 0 else np.nan,
            "MRL_Maximo_Promedio": float(np.mean(max_mrl)) if len(max_mrl) else np.nan,
        })

    return pd.DataFrame(summary_rows)


def plot_offset_summary_ci(offset_summary, window_name):
    """Grafica frecuencia vs offset promedio con IC 95%."""
    offset_summary = offset_summary.sort_values("Frecuencia_Hz").copy()
    x = offset_summary["Frecuencia_Hz"].to_numpy(dtype=float)
    y = offset_summary["Offset_ms_Promedio"].to_numpy(dtype=float)
    y_low = offset_summary["Offset_ms_IC95_Lower"].to_numpy(dtype=float)
    y_high = offset_summary["Offset_ms_IC95_Upper"].to_numpy(dtype=float)
    yerr = np.vstack([y - y_low, y_high - y])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.0,
        capsize=4,
        linewidth=1.5,
        markersize=4,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Offset promedio del maximo MRL (ms)")
    window_duration_s = offset_summary["Duracion_Ventana_s"].iloc[0]
    window_step_s = offset_summary["Salto_Ventana_s"].iloc[0]
    plt.title(
        f"Offset promedio por frecuencia con IC 95% - {window_name} "
        f"(ventana {window_duration_s:g}s, salto {window_step_s:g}s)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_6s_offset_mean_ci.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_lagged_mrl_6s_offsets(signal_a, signal_b, fs, window_name, start_s, end_s,
                                  center_freqs=None, window_width_hz=2.0,
                                  lags_ms=None, short_window_s=6.0,
                                  short_window_step_s=3.0):
    """Divide un tramo en ventanas cortas y resume el offset del MRL maximo."""
    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)
    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1, dtype=float)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    start_idx = int(round(start_s * fs))
    end_idx = int(round(end_s * fs))
    start_idx = max(0, min(start_idx, len(signal_a)))
    end_idx = max(start_idx + 1, min(end_idx, len(signal_a)))

    short_window_samples = int(round(short_window_s * fs))
    if short_window_samples <= 0:
        raise ValueError("La duracion de la ventana corta debe ser mayor que cero")

    short_window_step_samples = int(round(short_window_step_s * fs))
    if short_window_step_samples <= 0:
        raise ValueError("El salto de la ventana corta debe ser mayor que cero")

    total_samples = end_idx - start_idx
    if total_samples < short_window_samples:
        raise ValueError("El tramo seleccionado no alcanza para una ventana de 6 segundos")

    window_starts = np.arange(
        0,
        total_samples - short_window_samples + 1,
        short_window_step_samples,
        dtype=int,
    )
    segment_a = signal_a[start_idx:end_idx]
    segment_b = signal_b[start_idx:end_idx]

    mrl_rows = []
    maxima_rows = []
    half_width = window_width_hz / 2.0

    for center_freq in center_freqs:
        lowcut = max(0.0, center_freq - half_width)
        highcut = min(30.0, center_freq + half_width)
        freq_label = f"{int(center_freq)}Hz({int(lowcut)}-{int(highcut)}Hz)"

        filtered_a = butterworth_bandpass(segment_a, fs, float(lowcut), float(highcut), order=4)
        filtered_b = butterworth_bandpass(segment_b, fs, float(lowcut), float(highcut), order=4)
        phase_a = np.angle(hilbert(filtered_a))
        phase_b = np.angle(hilbert(filtered_b))

        for short_idx, win_start in enumerate(window_starts):
            win_end = win_start + short_window_samples
            phase_a_win = phase_a[win_start:win_end]
            phase_b_win = phase_b[win_start:win_end]
            mrl_values = compute_lagged_mrl_curve_from_phase(phase_a_win, phase_b_win, fs, lags_ms)

            window_start_s = start_s + win_start / fs
            window_end_s = window_start_s + short_window_s
            for lag_ms, mrl in zip(lags_ms, mrl_values):
                mrl_rows.append((
                    window_name,
                    short_idx + 1,
                    window_start_s,
                    window_end_s,
                    float(short_window_s),
                    float(short_window_step_s),
                    float(center_freq),
                    freq_label,
                    float(lag_ms),
                    float(mrl),
                ))

            if np.all(np.isnan(mrl_values)):
                max_lag = np.nan
                max_mrl = np.nan
            else:
                max_idx = int(np.nanargmax(mrl_values))
                max_lag = float(lags_ms[max_idx])
                max_mrl = float(mrl_values[max_idx])

            maxima_rows.append({
                "Segmento": window_name,
                "Ventana_6s": short_idx + 1,
                "Inicio_s": window_start_s,
                "Fin_s": window_end_s,
                "Duracion_Ventana_s": float(short_window_s),
                "Salto_Ventana_s": float(short_window_step_s),
                "Frecuencia_Hz": float(center_freq),
                "Ventana_Frecuencia": freq_label,
                "Offset_ms_Max_MRL": max_lag,
                "MRL_Maximo": max_mrl,
            })

    mrl_curves = pd.DataFrame(
        mrl_rows,
        columns=[
            "Segmento",
            "Ventana_6s",
            "Inicio_s",
            "Fin_s",
            "Duracion_Ventana_s",
            "Salto_Ventana_s",
            "Frecuencia_Hz",
            "Ventana_Frecuencia",
            "Lag_ms",
            "MRL",
        ],
    )
    offset_maxima = pd.DataFrame(maxima_rows)
    offset_summary = summarize_offset_maxima(offset_maxima)

    return mrl_curves, offset_maxima, offset_summary


def save_lagged_mrl_6s_outputs(signal_a, signal_b, fs, window_name, start_s, end_s,
                               center_freqs=None, window_width_hz=2.0,
                               lags_ms=None, short_window_s=6.0,
                               short_window_step_s=3.0):
    """Calcula, guarda y grafica offsets MRL en ventanas de 6 segundos."""
    mrl_curves, offset_maxima, offset_summary = compute_lagged_mrl_6s_offsets(
        signal_a,
        signal_b,
        fs,
        window_name,
        start_s,
        end_s,
        center_freqs=center_freqs,
        window_width_hz=window_width_hz,
        lags_ms=lags_ms,
        short_window_s=short_window_s,
        short_window_step_s=short_window_step_s,
    )

    mrl_curves.to_csv(f"mrl_lagged_{window_name}_6s_mrl.csv", index=False, encoding="utf-8-sig")
    offset_maxima.to_csv(f"mrl_lagged_{window_name}_6s_offset_maxima.csv", index=False, encoding="utf-8-sig")
    offset_summary.to_csv(f"mrl_lagged_{window_name}_6s_offset_summary_ci.csv", index=False, encoding="utf-8-sig")
    plot_offset_summary_ci(offset_summary, window_name)

    print(
        "Analisis 6s guardado para "
        f"{window_name}: mrl_lagged_{window_name}_6s_mrl.csv, "
        f"mrl_lagged_{window_name}_6s_offset_maxima.csv, "
        f"mrl_lagged_{window_name}_6s_offset_summary_ci.csv, "
        f"mrl_lagged_{window_name}_6s_offset_mean_ci.png "
        f"(ventana {short_window_s:g}s, salto {short_window_step_s:g}s)"
    )

    return mrl_curves, offset_maxima, offset_summary


def format_seconds_tag(value_s):
    """Convierte segundos a una etiqueta corta para nombres de archivo."""
    value_s = float(value_s)
    if value_s.is_integer():
        return f"{int(value_s)}s"
    return f"{value_s:g}s".replace(".", "p")


def compute_lagged_mrl_offsets_from_start_pairs(signal_a, signal_b, fs, window_name, start_s, end_s,
                                                starts_a_samples, starts_b_samples, analysis_type,
                                                center_freqs=None, window_width_hz=2.0,
                                                lags_ms=None, short_window_s=5.0,
                                                short_window_step_s=3.0):
    """Calcula offsets MRL usando pares de ventanas definidos por sus inicios."""
    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)
    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1, dtype=float)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)
    starts_a_samples = np.asarray(starts_a_samples, dtype=int)
    starts_b_samples = np.asarray(starts_b_samples, dtype=int)

    if len(starts_a_samples) != len(starts_b_samples):
        raise ValueError("Las listas de ventanas aleatorias deben tener el mismo largo")

    start_idx = int(round(start_s * fs))
    end_idx = int(round(end_s * fs))
    start_idx = max(0, min(start_idx, len(signal_a)))
    end_idx = max(start_idx + 1, min(end_idx, len(signal_a)))

    short_window_samples = int(round(short_window_s * fs))
    if short_window_samples <= 0:
        raise ValueError("La duracion de la ventana corta debe ser mayor que cero")

    total_samples = end_idx - start_idx
    if total_samples < short_window_samples:
        raise ValueError("El tramo seleccionado no alcanza para una ventana corta")

    max_start = total_samples - short_window_samples
    if np.any(starts_a_samples < 0) or np.any(starts_b_samples < 0):
        raise ValueError("Los inicios de ventana no pueden ser negativos")
    if np.any(starts_a_samples > max_start) or np.any(starts_b_samples > max_start):
        raise ValueError("Al menos una ventana queda fuera del periodo seleccionado")

    segment_a = signal_a[start_idx:end_idx]
    segment_b = signal_b[start_idx:end_idx]

    mrl_rows = []
    maxima_rows = []
    half_width = window_width_hz / 2.0

    for center_freq in center_freqs:
        lowcut = max(0.0, center_freq - half_width)
        highcut = min(30.0, center_freq + half_width)
        freq_label = f"{int(center_freq)}Hz({int(lowcut)}-{int(highcut)}Hz)"

        filtered_a = butterworth_bandpass(segment_a, fs, float(lowcut), float(highcut), order=4)
        filtered_b = butterworth_bandpass(segment_b, fs, float(lowcut), float(highcut), order=4)
        phase_a = np.angle(hilbert(filtered_a))
        phase_b = np.angle(hilbert(filtered_b))

        for window_idx, (start_a, start_b) in enumerate(zip(starts_a_samples, starts_b_samples)):
            end_a = start_a + short_window_samples
            end_b = start_b + short_window_samples
            phase_a_win = phase_a[start_a:end_a]
            phase_b_win = phase_b[start_b:end_b]
            mrl_values = compute_lagged_mrl_curve_from_phase(phase_a_win, phase_b_win, fs, lags_ms)

            start_a_s = start_s + start_a / fs
            start_b_s = start_s + start_b / fs
            end_a_s = start_a_s + short_window_s
            end_b_s = start_b_s + short_window_s

            for lag_ms, mrl in zip(lags_ms, mrl_values):
                mrl_rows.append((
                    window_name,
                    analysis_type,
                    window_idx + 1,
                    start_a_s,
                    end_a_s,
                    start_b_s,
                    end_b_s,
                    float(short_window_s),
                    float(short_window_step_s),
                    float(center_freq),
                    freq_label,
                    float(lag_ms),
                    float(mrl),
                ))

            if np.all(np.isnan(mrl_values)):
                max_lag = np.nan
                max_mrl = np.nan
            else:
                max_idx = int(np.nanargmax(mrl_values))
                max_lag = float(lags_ms[max_idx])
                max_mrl = float(mrl_values[max_idx])

            maxima_rows.append({
                "Segmento": window_name,
                "Tipo": analysis_type,
                "Ventana_ID": window_idx + 1,
                "Inicio_A_s": start_a_s,
                "Fin_A_s": end_a_s,
                "Inicio_B_s": start_b_s,
                "Fin_B_s": end_b_s,
                "Duracion_Ventana_s": float(short_window_s),
                "Salto_Ventana_s": float(short_window_step_s),
                "Frecuencia_Hz": float(center_freq),
                "Ventana_Frecuencia": freq_label,
                "Offset_ms_Max_MRL": max_lag,
                "MRL_Maximo": max_mrl,
            })

    mrl_curves = pd.DataFrame(
        mrl_rows,
        columns=[
            "Segmento",
            "Tipo",
            "Ventana_ID",
            "Inicio_A_s",
            "Fin_A_s",
            "Inicio_B_s",
            "Fin_B_s",
            "Duracion_Ventana_s",
            "Salto_Ventana_s",
            "Frecuencia_Hz",
            "Ventana_Frecuencia",
            "Lag_ms",
            "MRL",
        ],
    )
    offset_maxima = pd.DataFrame(maxima_rows)
    offset_summary = summarize_offset_maxima(offset_maxima)
    offset_summary.insert(0, "Tipo", analysis_type)

    return mrl_curves, offset_maxima, offset_summary


def plot_real_vs_random_offset_summary_ci(real_summary, random_summary, window_name, analysis_tag):
    """Grafica en la misma figura la curva real y la curva al azar con IC 95%."""
    plt.figure(figsize=(10, 6))
    plot_specs = [
        (real_summary.sort_values("Frecuencia_Hz"), "Real", "tab:blue", "o-"),
        (random_summary.sort_values("Frecuencia_Hz"), "Azar", "tab:orange", "s-"),
    ]

    for summary, label, color, fmt in plot_specs:
        x = summary["Frecuencia_Hz"].to_numpy(dtype=float)
        y = summary["Offset_ms_Promedio"].to_numpy(dtype=float)
        y_low = summary["Offset_ms_IC95_Lower"].to_numpy(dtype=float)
        y_high = summary["Offset_ms_IC95_Upper"].to_numpy(dtype=float)
        yerr = np.vstack([y - y_low, y_high - y])
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=fmt,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            linewidth=1.5,
            markersize=4,
            label=label,
            alpha=0.9,
        )

    window_duration_s = real_summary["Duracion_Ventana_s"].iloc[0]
    window_step_s = real_summary["Salto_Ventana_s"].iloc[0]
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Offset promedio del maximo MRL (ms)")
    plt.title(
        f"Offset promedio real vs azar con IC 95% - {window_name} "
        f"(ventana {window_duration_s:g}s, salto {window_step_s:g}s)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar_offset_mean_ci.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_lagged_mrl_random_comparison(signal_a, signal_b, fs, window_name, start_s, end_s,
                                      center_freqs=None, window_width_hz=2.0,
                                      lags_ms=None, short_window_s=5.0,
                                      short_window_step_s=3.0, random_seed=20260811):
    """Guarda la comparacion real vs azar para offsets de maximo MRL."""
    start_idx = int(round(start_s * fs))
    end_idx = int(round(end_s * fs))
    total_samples = max(0, end_idx - start_idx)
    short_window_samples = int(round(short_window_s * fs))
    short_window_step_samples = int(round(short_window_step_s * fs))
    if total_samples < short_window_samples:
        raise ValueError("El periodo seleccionado es demasiado corto para la ventana solicitada")

    aligned_starts = np.arange(
        0,
        total_samples - short_window_samples + 1,
        short_window_step_samples,
        dtype=int,
    )
    n_windows = len(aligned_starts)

    rng = np.random.default_rng(random_seed)
    max_start = total_samples - short_window_samples
    random_starts_a = rng.integers(0, max_start + 1, size=n_windows)
    random_starts_b = rng.integers(0, max_start + 1, size=n_windows)

    real_curves, real_maxima, real_summary = compute_lagged_mrl_offsets_from_start_pairs(
        signal_a,
        signal_b,
        fs,
        window_name,
        start_s,
        end_s,
        aligned_starts,
        aligned_starts,
        "Real",
        center_freqs=center_freqs,
        window_width_hz=window_width_hz,
        lags_ms=lags_ms,
        short_window_s=short_window_s,
        short_window_step_s=short_window_step_s,
    )

    random_curves, random_maxima, random_summary = compute_lagged_mrl_offsets_from_start_pairs(
        signal_a,
        signal_b,
        fs,
        window_name,
        start_s,
        end_s,
        random_starts_a,
        random_starts_b,
        "Azar",
        center_freqs=center_freqs,
        window_width_hz=window_width_hz,
        lags_ms=lags_ms,
        short_window_s=short_window_s,
        short_window_step_s=short_window_step_s,
    )

    analysis_tag = f"{format_seconds_tag(short_window_s)}_step{format_seconds_tag(short_window_step_s)}"
    real_curves.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_real_mrl.csv", index=False, encoding="utf-8-sig")
    real_maxima.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_real_offset_maxima.csv", index=False, encoding="utf-8-sig")
    real_summary.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_real_offset_summary_ci.csv", index=False, encoding="utf-8-sig")
    random_curves.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_azar_mrl.csv", index=False, encoding="utf-8-sig")
    random_maxima.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_azar_offset_maxima.csv", index=False, encoding="utf-8-sig")
    random_summary.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_azar_offset_summary_ci.csv", index=False, encoding="utf-8-sig")

    combined_summary = pd.concat([real_summary, random_summary], ignore_index=True)
    combined_summary.to_csv(f"mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar_offset_summary_ci.csv", index=False, encoding="utf-8-sig")
    plot_real_vs_random_offset_summary_ci(real_summary, random_summary, window_name, analysis_tag)

    print(
        f"Comparacion real vs azar guardada para {window_name}: "
        f"mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar_offset_mean_ci.png "
        f"(n={n_windows}, ventana {short_window_s:g}s, salto {short_window_step_s:g}s, seed={random_seed})"
    )

    return real_summary, random_summary

# ============================================================
# COMPARACIÓN 5 MINUTOS: REAL VS 50 ITERACIONES AL AZAR
# ============================================================

def save_lagged_mrl_5min_random_50_comparison(
    signal_a,
    signal_b,
    fs,
    window_name,
    pool_start_s,
    pool_end_s,
    real_start_s,
    real_end_s,
    center_freqs=None,
    window_width_hz=2.0,
    lags_ms=None,
    n_iterations=50,
    random_seed=20260814
):
    """
    Compara el offset del máximo MRL de una ventana real de 5 minutos
    contra el promedio de n_iterations ventanas aleatorias de 5 minutos.

    Para cada iteración aleatoria:
      - Se selecciona independientemente una ventana de 5 min de señal A.
      - Se selecciona independientemente una ventana de 5 min de señal B.
      - Se calcula el MRL para cada frecuencia y offset.
      - Se obtiene el offset donde el MRL es máximo.

    Las ventanas de A y B NO están sincronizadas temporalmente.
    """

    ventana_s = 300.0  # 5 minutos

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)

    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1, dtype=float)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    if real_end_s - real_start_s != ventana_s:
        raise ValueError(
            "La ventana REAL debe tener exactamente 5 minutos (300 s)."
        )

    if pool_end_s - pool_start_s < ventana_s:
        raise ValueError(
            "El periodo disponible para seleccionar ventanas aleatorias "
            "debe ser de al menos 5 minutos."
        )

    if real_start_s < pool_start_s or real_end_s > pool_end_s:
        raise ValueError(
            "La ventana real debe estar contenida dentro del periodo "
            "disponible para el muestreo aleatorio."
        )

    # --------------------------------------------------------
    # Convertir segundos a muestras
    # --------------------------------------------------------

    ventana_samples = int(round(ventana_s * fs))

    pool_start_idx = int(round(pool_start_s * fs))
    pool_end_idx = int(round(pool_end_s * fs))

    pool_start_idx = max(0, pool_start_idx)
    pool_end_idx = min(
        pool_end_idx,
        len(signal_a),
        len(signal_b)
    )

    pool_samples = pool_end_idx - pool_start_idx

    if pool_samples < ventana_samples:
        raise ValueError(
            "El periodo disponible no contiene una ventana completa de 5 minutos."
        )

    max_random_offset = pool_samples - ventana_samples

    # --------------------------------------------------------
    # Generador aleatorio reproducible
    # --------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    # ========================================================
    # 1. ANÁLISIS REAL
    # ========================================================

    # El inicio de la ventana real se expresa respecto al
    # inicio del periodo disponible.
    real_start_relative = int(
        round((real_start_s - pool_start_s) * fs)
    )

    real_end_relative = real_start_relative + ventana_samples

    (
        real_curves,
        real_maxima,
        real_summary
    ) = compute_lagged_mrl_offsets_from_start_pairs(
        signal_a,
        signal_b,
        fs,
        window_name,
        pool_start_s,
        pool_end_s,
        np.array([real_start_relative]),
        np.array([real_start_relative]),
        "Real_5min",
        center_freqs=center_freqs,
        window_width_hz=window_width_hz,
        lags_ms=lags_ms,
        short_window_s=ventana_s,
        short_window_step_s=ventana_s
    )

    # ========================================================
    # 2. 50 ITERACIONES ALEATORIAS
    # ========================================================

    random_maxima_all = []
    random_curves_all = []

    for iteration in range(1, n_iterations + 1):

        # IMPORTANTE:
        # A y B reciben posiciones aleatorias INDEPENDIENTES.
        random_start_a = int(
            rng.integers(0, max_random_offset + 1)
        )

        random_start_b = int(
            rng.integers(0, max_random_offset + 1)
        )

        (
            random_curves,
            random_maxima,
            _
        ) = compute_lagged_mrl_offsets_from_start_pairs(
            signal_a,
            signal_b,
            fs,
            window_name,
            pool_start_s,
            pool_end_s,
            np.array([random_start_a]),
            np.array([random_start_b]),
            "Azar_5min",
            center_freqs=center_freqs,
            window_width_hz=window_width_hz,
            lags_ms=lags_ms,
            short_window_s=ventana_s,
            short_window_step_s=ventana_s
        )

        # Identificar explícitamente la iteración
        random_maxima = random_maxima.copy()
        random_maxima.insert(0, "Iteracion", iteration)

        random_curves = random_curves.copy()
        random_curves.insert(0, "Iteracion", iteration)

        random_maxima_all.append(random_maxima)
        random_curves_all.append(random_curves)

    # --------------------------------------------------------
    # Unir las 50 iteraciones
    # --------------------------------------------------------

    random_maxima_all = pd.concat(
        random_maxima_all,
        ignore_index=True
    )

    random_curves_all = pd.concat(
        random_curves_all,
        ignore_index=True
    )

    # ========================================================
    # 3. RESUMEN DEL AZAR
    # ========================================================

    random_summary = summarize_offset_maxima(
        random_maxima_all
    )

    random_summary.insert(
        0,
        "Tipo",
        "Azar_5min_50_iteraciones"
    )

    # ========================================================
    # 4. GUARDAR RESULTADOS
    # ========================================================

    analysis_tag = "5min_50iter"

    real_curves.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_real_mrl.csv",
        index=False,
        encoding="utf-8-sig"
    )

    real_maxima.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_real_offset_maxima.csv",
        index=False,
        encoding="utf-8-sig"
    )

    random_curves_all.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_azar_50iter_mrl.csv",
        index=False,
        encoding="utf-8-sig"
    )

    random_maxima_all.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_azar_50iter_offset_maxima.csv",
        index=False,
        encoding="utf-8-sig"
    )

    random_summary.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_azar_promedio_ci.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 5. COMPARACIÓN REAL VS AZAR
    # ========================================================

    # Para la medición real tenemos una sola ventana.
    # Por lo tanto, usamos directamente sus offsets máximos.
    real_comparison = real_maxima[
        [
            "Frecuencia_Hz",
            "Ventana_Frecuencia",
            "Offset_ms_Max_MRL",
            "MRL_Maximo"
        ]
    ].copy()

    real_comparison = real_comparison.rename(
        columns={
            "Offset_ms_Max_MRL": "Offset_Real_ms",
            "MRL_Maximo": "MRL_Real"
        }
    )

    random_comparison = random_summary[
        [
            "Frecuencia_Hz",
            "Ventana_Frecuencia",
            "Offset_ms_Promedio",
            "Offset_ms_IC95_Lower",
            "Offset_ms_IC95_Upper",
            "MRL_Maximo_Promedio"
        ]
    ].copy()

    random_comparison = random_comparison.rename(
        columns={
            "Offset_ms_Promedio": "Offset_Azar_Promedio_ms",
            "Offset_ms_IC95_Lower": "Offset_Azar_IC95_Lower",
            "Offset_ms_IC95_Upper": "Offset_Azar_IC95_Upper",
            "MRL_Maximo_Promedio": "MRL_Azar_Promedio"
        }
    )

    comparison = pd.merge(
        real_comparison,
        random_comparison,
        on=["Frecuencia_Hz", "Ventana_Frecuencia"],
        how="outer"
    )

    comparison.to_csv(
        f"mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 6. GRÁFICO FINAL
    # ========================================================

    comparison = comparison.sort_values(
        "Frecuencia_Hz"
    )

    x = comparison["Frecuencia_Hz"].to_numpy(dtype=float)

    y_real = comparison["Offset_Real_ms"].to_numpy(dtype=float)

    y_random = comparison[
        "Offset_Azar_Promedio_ms"
    ].to_numpy(dtype=float)

    y_low = comparison[
        "Offset_Azar_IC95_Lower"
    ].to_numpy(dtype=float)

    y_high = comparison[
        "Offset_Azar_IC95_Upper"
    ].to_numpy(dtype=float)

    # Error asimétrico del azar
    yerr_random = np.vstack([
        y_random - y_low,
        y_high - y_random
    ])

    plt.figure(figsize=(11, 6))

    # Real: una única curva, sin IC porque n=1
    plt.plot(
        x,
        y_real,
        "o-",
        linewidth=2,
        markersize=4,
        label="Real - ventana de 5 min"
    )

    # Azar: promedio de 50 iteraciones + IC95%
    plt.errorbar(
        x,
        y_random,
        yerr=yerr_random,
        fmt="s-",
        linewidth=1.5,
        markersize=4,
        capsize=3,
        label="Azar - promedio de 50 iteraciones"
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Offset del máximo MRL (ms)")

    plt.title(
        f"Offset del máximo MRL: Real vs Azar - {window_name}\n"
        f"Ventanas de 5 minutos | 50 iteraciones aleatorias | IC 95%"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 7. INFORMACIÓN POR CONSOLA
    # ========================================================

    print("\n==============================================")
    print("COMPARACIÓN REAL VS AZAR - 5 MINUTOS")
    print("==============================================")
    print(f"Segmento: {window_name}")
    print(f"Ventana real: {real_start_s} - {real_end_s} s")
    print(f"Periodo disponible para azar: {pool_start_s} - {pool_end_s} s")
    print(f"Iteraciones aleatorias: {n_iterations}")
    print(f"Duración de cada ventana: {ventana_s} s")
    print(f"Seed aleatoria: {random_seed}")

    print("\nResultados guardados:")
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_real_mrl.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_real_offset_maxima.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_azar_50iter_mrl.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_azar_50iter_offset_maxima.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_azar_promedio_ci.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar.csv"
    )
    print(
        f"- mrl_lagged_{window_name}_{analysis_tag}_real_vs_azar.png"
    )

    return (
        real_maxima,
        random_maxima_all,
        random_summary,
        comparison
    )
# ============================================================
# COMPARACIÓN 11-16 MIN VS 3 PERIODOS ALEATORIOS DE 5 MIN
# ============================================================

def save_lagged_mrl_5min_three_random_comparison(
    signal_a,
    signal_b,
    fs,
    window_name,
    real_start_s,
    real_end_s,
    pool_start_s=0.0,
    pool_end_s=None,
    center_freqs=None,
    window_width_hz=2.0,
    lags_ms=None,
    random_seed=20260817
):
    """
    Compara una ventana REAL de 5 minutos contra
    3 ventanas aleatorias de 5 minutos.

    Para cada periodo aleatorio:
        - Se selecciona independientemente una ventana de 5 min
          para la señal A.
        - Se selecciona independientemente una ventana de 5 min
          para la señal B.
        - A y B NO están sincronizadas temporalmente.
        - Se calcula MRL para cada frecuencia y offset.
        - Se obtiene el offset donde el MRL es máximo.

    Las 3 ventanas aleatorias:
        - no pueden coincidir con la ventana real;
        - no pueden solaparse entre sí;
        - tienen exactamente 5 minutos.
    """

    ventana_s = 300.0  # 5 minutos

    # --------------------------------------------------------
    # Configuración por defecto
    # --------------------------------------------------------

    if center_freqs is None:
        center_freqs = np.arange(1, 30, dtype=float)

    if lags_ms is None:
        lags_ms = np.arange(-250, 251, 1, dtype=float)

    center_freqs = np.asarray(center_freqs, dtype=float)
    lags_ms = np.asarray(lags_ms, dtype=float)

    if pool_end_s is None:
        pool_end_s = len(signal_a) / fs

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    if abs((real_end_s - real_start_s) - ventana_s) > 1e-6:
        raise ValueError(
            "La ventana REAL debe tener exactamente 5 minutos."
        )

    if pool_end_s - pool_start_s < ventana_s:
        raise ValueError(
            "El periodo disponible es menor que 5 minutos."
        )

    if real_start_s < pool_start_s or real_end_s > pool_end_s:
        raise ValueError(
            "La ventana real debe estar dentro del periodo disponible."
        )

    # --------------------------------------------------------
    # Función para comprobar solapamiento
    # --------------------------------------------------------

    def se_solapan(inicio1, fin1, inicio2, fin2):
        return max(inicio1, inicio2) < min(fin1, fin2)

    # --------------------------------------------------------
    # Generador aleatorio
    # --------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    max_start = pool_end_s - ventana_s

    # --------------------------------------------------------
    # Buscar 3 ventanas aleatorias NO solapadas
    # --------------------------------------------------------

    random_windows = []

    intentos = 0
    max_intentos = 100000

    while len(random_windows) < 2:

        intentos += 1

        if intentos > max_intentos:
            raise RuntimeError(
                "No fue posible encontrar 3 ventanas aleatorias "
                "de 5 minutos que no se solapen."
            )

        candidato = float(
            rng.uniform(pool_start_s, max_start)
        )

        candidato_fin = candidato + ventana_s

        # No puede coincidir con la ventana real
        if se_solapan(
            candidato,
            candidato_fin,
            real_start_s,
            real_end_s
        ):
            continue

        # No puede coincidir con ninguna ventana aleatoria anterior
        conflicto = False

        for inicio_existente, fin_existente in random_windows:

            if se_solapan(
                candidato,
                candidato_fin,
                inicio_existente,
                fin_existente
            ):
                conflicto = True
                break

        if conflicto:
            continue

        random_windows.append(
            (candidato, candidato_fin)
        )

    # Ordenar cronológicamente
    random_windows.sort()

    print("\n==============================================")
    print("COMPARACIÓN REAL VS 3 PERIODOS ALEATORIOS")
    print("==============================================")

    print(
        f"Periodo REAL: "
        f"{real_start_s/60:.2f} - {real_end_s/60:.2f} min"
    )

    for i, (inicio, fin) in enumerate(random_windows, 1):

        print(
            f"Azar {i}: "
            f"{inicio/60:.2f} - {fin/60:.2f} min"
        )

    # --------------------------------------------------------
    # Calcular REAL
    # --------------------------------------------------------

    real_start_idx = int(round(real_start_s * fs))
    real_end_idx = int(round(real_end_s * fs))

    real_a = signal_a[
        real_start_idx:real_end_idx
    ]

    real_b = signal_b[
        real_start_idx:real_end_idx
    ]

    # IMPORTANTE:
    # Utilizamos la función existente de tu pipeline
    # para calcular MRL y obtener el offset máximo.

    real_df = compute_lagged_mrl_by_window(
        signal_a,
        signal_b,
        fs,
        center_freqs=center_freqs,
        window_width_hz=window_width_hz,
        lags_ms=lags_ms,
        start_s=real_start_s,
        end_s=real_end_s
    )

    # --------------------------------------------------------
    # Obtener offset máximo REAL por frecuencia
    # --------------------------------------------------------

    real_maximos = (
        real_df
        .loc[
            real_df.groupby("Ventana")["MRL"].idxmax()
        ]
        .copy()
    )

    real_maximos["Tipo"] = "Real"
    real_maximos["Periodo"] = (
        f"{real_start_s/60:.2f}-{real_end_s/60:.2f} min"
    )

    # --------------------------------------------------------
    # Calcular las 3 ventanas aleatorias
    # --------------------------------------------------------

    random_results = []

    for i, (inicio_s, fin_s) in enumerate(
        random_windows,
        start=1
    ):

        inicio_idx = int(round(inicio_s * fs))
        fin_idx = int(round(fin_s * fs))

        # Señal A
        random_a = signal_a[
            inicio_idx:fin_idx
        ]

        # Señal B
        random_b = signal_b[
            inicio_idx:fin_idx
        ]

        # ----------------------------------------------------
        # IMPORTANTE:
        # A y B se seleccionan del MISMO periodo aleatorio
        # en esta versión.
        # ----------------------------------------------------

        df_random = compute_lagged_mrl_by_window(
            signal_a,
            signal_b,
            fs,
            center_freqs=center_freqs,
            window_width_hz=window_width_hz,
            lags_ms=lags_ms,
            start_s=inicio_s,
            end_s=fin_s
        )

        maximos_random = (
            df_random
            .loc[
                df_random.groupby("Ventana")["MRL"].idxmax()
            ]
            .copy()
        )

        maximos_random["Tipo"] = f"Azar {i}"

        maximos_random["Periodo"] = (
            f"{inicio_s/60:.2f}-{fin_s/60:.2f} min"
        )

        random_results.append(
            maximos_random
        )

    # --------------------------------------------------------
    # Unir resultados
    # --------------------------------------------------------

    todos = pd.concat(
        [real_maximos] + random_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    archivo_csv = (
        f"mrl_lagged_{window_name}"
        "_real_vs_3_azar_offset_maxima.csv"
    )

    todos.to_csv(
        archivo_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Crear tabla para gráfico
    # --------------------------------------------------------

    plt.figure(figsize=(12, 7))

    tipos = [
        "Real",
        "Azar 1",
        "Azar 2",
        "Azar 3"
    ]

    for tipo in tipos:

        datos_tipo = todos[
            todos["Tipo"] == tipo
        ]

        # Agrupar por frecuencia
        curva = (
            datos_tipo
            .groupby("Frecuencia_Hz", as_index=False)
            ["Lag_ms"]
            .mean()
        )

        plt.plot(
            curva["Frecuencia_Hz"],
            curva["Lag_ms"],
            marker="o",
            linewidth=2,
            label=tipo
        )

    # --------------------------------------------------------
    # Configuración del gráfico
    # --------------------------------------------------------

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Frecuencia central (Hz)"
    )

    plt.ylabel(
        "Offset del MRL máximo (ms)"
    )

    plt.title(
        "Offset del MRL máximo: "
        "11–16 min vs 3 periodos aleatorios"
    )

    plt.ylim(-250, 250)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    archivo_figura = (
        f"mrl_lagged_{window_name}"
        "_real_vs_3_azar_offset_maxima.png"
    )

    plt.savefig(
        archivo_figura,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nArchivos generados:")
    print(
        f"  {archivo_csv}"
    )
    print(
        f"  {archivo_figura}"
    )

    return todos, random_windows

# ============================================================
# COMPARACIÓN REAL VS AZAR - VENTANAS DE 5 MINUTOS
# ============================================================

def save_5min_real_vs_random_comparison(
    signal_a,
    signal_b,
    fs,
    window_name,
    start_s,
    end_s,
    n_random=100,
    window_s=300.0,
    center_freqs=np.arange(1, 30, dtype=float),
    window_width_hz=2.0,
    lags_ms=np.arange(-250, 251, 1, dtype=float),
    random_seed=20260813
):
    """
    Compara el offset promedio por frecuencia de una ventana REAL
    de 5 minutos contra el promedio de múltiples ventanas ALEATORIAS
    de 5 minutos.

    Para cada ventana se calcula el MRL y posteriormente el offset
    correspondiente a cada frecuencia.

    Parámetros
    ----------
    signal_a : np.ndarray
        Primera señal.

    signal_b : np.ndarray
        Segunda señal.

    fs : float
        Frecuencia de muestreo.

    window_name : str
        Nombre del período analizado.

    start_s : float
        Inicio del período de referencia.

    end_s : float
        Fin del período de referencia.

    n_random : int
        Número de ventanas aleatorias.

    window_s : float
        Duración de cada ventana en segundos.
        Por defecto: 300 s = 5 minutos.

    center_freqs : np.ndarray
        Frecuencias centrales.

    window_width_hz : float
        Ancho de cada ventana espectral.

    lags_ms : np.ndarray
        Retardos utilizados para calcular MRL.

    random_seed : int
        Semilla para reproducibilidad.
    """

    print("\n" + "=" * 70)
    print("COMPARACIÓN REAL VS AZAR - VENTANAS DE 5 MINUTOS")
    print("=" * 70)

    print(f"Período: {window_name}")
    print(f"Ventana real: {window_s / 60:.1f} minutos")
    print(f"Ventanas aleatorias: {n_random}")

    # --------------------------------------------------------
    # 1. Determinar cantidad de muestras
    # --------------------------------------------------------

    window_samples = int(round(window_s * fs))

    total_samples = min(len(signal_a), len(signal_b))

    if total_samples < window_samples:
        raise ValueError(
            f"No hay suficientes datos para una ventana de "
            f"{window_s / 60:.1f} minutos."
        )

    # --------------------------------------------------------
    # 2. Ventana REAL
    # --------------------------------------------------------

    real_start = int(round(start_s * fs))
    real_end = real_start + window_samples

    if real_end > total_samples:
        raise ValueError(
            "La ventana real de 5 minutos supera el tamaño "
            "disponible del registro."
        )

    real_a = signal_a[real_start:real_end]
    real_b = signal_b[real_start:real_end]

    print(
        f"Ventana real: "
        f"{start_s:.1f} s → {(start_s + window_s):.1f} s"
    )

    # --------------------------------------------------------
    # 3. Función interna para calcular offset
    # --------------------------------------------------------

    def calcular_offset(signal_a_window, signal_b_window):

        resultados_offset = []

        for f0 in center_freqs:

            # Límites de la banda
            f_low = f0 - window_width_hz / 2
            f_high = f0 + window_width_hz / 2

            # Aquí utilizamos el mismo cálculo de MRL
            # que emplea tu análisis existente.
            #
            # IMPORTANTE:
            # Esta parte debe llamar a la misma función que ya
            # utilizas para obtener MRL en tu pipeline.

            mrl_por_lag = calcular_mrl_por_lag(
                signal_a_window,
                signal_b_window,
                fs,
                f_low,
                f_high,
                lags_ms
            )

            # Buscar el máximo MRL
            indice_max = np.argmax(mrl_por_lag)

            offset = lags_ms[indice_max]

            resultados_offset.append(offset)

        return np.asarray(resultados_offset)

    # --------------------------------------------------------
    # 4. Calcular OFFSET REAL
    # --------------------------------------------------------

    print("\nCalculando ventana real...")

    offset_real = calcular_offset(
        real_a,
        real_b
    )

    # --------------------------------------------------------
    # 5. Generar ventanas ALEATORIAS
    # --------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    max_start = total_samples - window_samples

    offsets_random = []

    print("\nCalculando ventanas aleatorias...")

    for i in range(n_random):

        random_start = rng.integers(
            0,
            max_start + 1
        )

        random_end = random_start + window_samples

        random_a = signal_a[
            random_start:random_end
        ]

        random_b = signal_b[
            random_start:random_end
        ]

        offset_random = calcular_offset(
            random_a,
            random_b
        )

        offsets_random.append(
            offset_random
        )

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Ventana aleatoria "
                f"{i + 1}/{n_random}"
            )

    offsets_random = np.asarray(offsets_random)

    # --------------------------------------------------------
    # 6. Promedio de los promedios aleatorios
    # --------------------------------------------------------

    random_mean = np.mean(
        offsets_random,
        axis=0
    )

    random_std = np.std(
        offsets_random,
        axis=0,
        ddof=1
    )

    # --------------------------------------------------------
    # 7. IC95% REAL
    # --------------------------------------------------------

    # Como tenemos una única ventana real, no podemos estimar
    # un IC entre múltiples ventanas reales directamente.
    #
    # Por eso utilizamos bootstrap sobre los datos de la ventana
    # real.

    n_bootstrap = 1000

    rng_boot = np.random.default_rng(
        random_seed + 1000
    )

    bootstrap_offsets = []

    n_samples_real = len(real_a)

    for _ in range(n_bootstrap):

        # Muestreo con reemplazo
        indices = rng_boot.integers(
            0,
            n_samples_real,
            size=n_samples_real
        )

        boot_a = real_a[indices]
        boot_b = real_b[indices]

        boot_offset = calcular_offset(
            boot_a,
            boot_b
        )

        bootstrap_offsets.append(
            boot_offset
        )

    bootstrap_offsets = np.asarray(
        bootstrap_offsets
    )

    real_ci_low = np.percentile(
        bootstrap_offsets,
        2.5,
        axis=0
    )

    real_ci_high = np.percentile(
        bootstrap_offsets,
        97.5,
        axis=0
    )

    # --------------------------------------------------------
    # 8. IC95% AZAR
    # --------------------------------------------------------

    n = offsets_random.shape[0]

    random_sem = (
        random_std /
        np.sqrt(n)
    )

    t_crit = t.ppf(
        0.975,
        df=n - 1
    )

    random_ci_low = (
        random_mean -
        t_crit * random_sem
    )

    random_ci_high = (
        random_mean +
        t_crit * random_sem
    )

    # --------------------------------------------------------
    # 9. Crear DataFrame
    # --------------------------------------------------------

    resultados = pd.DataFrame({

        "Frecuencia_Hz": center_freqs,

        "Offset_Real_ms": offset_real,

        "IC95_Real_Inferior_ms":
            real_ci_low,

        "IC95_Real_Superior_ms":
            real_ci_high,

        "Offset_Azar_Promedio_ms":
            random_mean,

        "IC95_Azar_Inferior_ms":
            random_ci_low,

        "IC95_Azar_Superior_ms":
            random_ci_high
    })

    # --------------------------------------------------------
    # 10. Guardar CSV
    # --------------------------------------------------------

    nombre_csv = (
        f"Real_vs_Azar_5min_{window_name}.csv"
    )

    resultados.to_csv(
        nombre_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCSV guardado como: {nombre_csv}"
    )

    # --------------------------------------------------------
    # 11. Gráfico
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    # ---------------- REAL ----------------

    plt.plot(
        center_freqs,
        offset_real,
        linewidth=2.5,
        label="Real"
    )

    plt.fill_between(
        center_freqs,
        real_ci_low,
        real_ci_high,
        alpha=0.20,
        label="IC95% Real"
    )

    # ---------------- AZAR ----------------

    plt.plot(
        center_freqs,
        random_mean,
        linewidth=2.5,
        linestyle="--",
        label="Azar"
    )

    plt.fill_between(
        center_freqs,
        random_ci_low,
        random_ci_high,
        alpha=0.20,
        label="IC95% Azar"
    )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=1
    )

    plt.xlabel(
        "Frecuencia central (Hz)"
    )

    plt.ylabel(
        "Offset de MRL (ms)"
    )

    plt.title(
        f"Offset promedio por frecuencia\n"
        f"Real vs. azar - {window_name} - "
        f"ventanas de 5 minutos"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    nombre_png = (
        f"Real_vs_Azar_5min_{window_name}.png"
    )

    plt.savefig(
        nombre_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Gráfico guardado como: {nombre_png}"
    )

    # --------------------------------------------------------
    # 12. Información resumen
    # --------------------------------------------------------

    print("\nResumen:")
    print(
        f"  Ventana real: {window_s / 60:.1f} minutos"
    )
    print(
        f"  Ventanas aleatorias: {n_random}"
    )
    print(
        f"  Bootstrap real: {n_bootstrap}"
    )

    return resultados

import numpy as np
import pandas as pd

def save_lagged_mrl_5min_random_comparison(
    signal_a,
    signal_b,
    fs,
    offsets_s,
    real_start_s,
    total_duration_s,
    window_duration_s=300.0,  # 5 minutos = 300 segundos
    n_random_samples=100,
    random_seed=42,
    output_csv_path="mrl_5min_comparison.csv",
    output_fig_path="mrl_5min_comparison.png"
):
    """
    Calcula el MRL con desfases (lagged MRL) para una ventana real de 5 minutos
    y la compara contra el promedio e IC 95% de N ventanas aleatorias de 5 minutos.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # -------------------------------------------------------------------------
    # 1. Medición REAL (Ventana fija de 5 minutos)
    # -------------------------------------------------------------------------
    # Definimos el par de inicio para la ventana real elegida (ej. min 1 a 6 -> real_start_s = 60.0)
    real_start_pairs = [(real_start_s, real_start_s)]
    
    # MRL real para los diferentes offsets
    # Retorna un array o dict con los valores de MRL por offset
    real_mrl_curve = compute_lagged_mrl_offsets_from_start_pairs(
        signal_a=signal_a,
        signal_b=signal_b,
        fs=fs,
        start_pairs=real_start_pairs,
        window_s=window_duration_s,
        offsets_s=offsets_s
    )
    
    # Asegurar que sea array de numpy 1D
    real_mrl_curve = np.array(real_mrl_curve).squeeze()

    # -------------------------------------------------------------------------
    # 2. Medición AZAR (N ventanas aleatorias de 5 minutos)
    # -------------------------------------------------------------------------
    max_start_s = total_duration_s - window_duration_s
    if max_start_s <= 0:
        raise ValueError("La duración total del registro es menor que el tamaño de ventana (5 min).")

    random_curves = []

    for _ in range(n_random_samples):
        # Tomar un inicio aleatorio válido dentro del registro
        rand_start_a = np.random.uniform(0, max_start_s)
        rand_start_b = np.random.uniform(0, max_start_s)  # O usar rand_start_a si ambas señales deben iniciar sincronizadas al azar

        rand_start_pairs = [(rand_start_a, rand_start_b)]

        # Calcular MRL para esta ventana aleatoria
        mrl_rand = compute_lagged_mrl_offsets_from_start_pairs(
            signal_a=signal_a,
            signal_b=signal_b,
            fs=fs,
            start_pairs=rand_start_pairs,
            window_s=window_duration_s,
            offsets_s=offsets_s
        )
        random_curves.append(np.array(mrl_rand).squeeze())

    # Convertir a matriz 2D: (n_random_samples, n_offsets)
    random_curves = np.array(random_curves)

    # -------------------------------------------------------------------------
    # 3. Cálculo de Estadísticas para el Azar (Promedio e Intervalo de Confianza 95%)
    # -------------------------------------------------------------------------
    mean_random = np.mean(random_curves, axis=0)
    
    # Percentiles 2.5% y 97.5% para IC del 95% no paramétrico
    ci_lower = np.percentile(random_curves, 2.5, axis=0)
    ci_upper = np.percentile(random_curves, 97.5, axis=0)

    # -------------------------------------------------------------------------
    # 4. Exportar Resultados a CSV
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame({
        'offset_s': offsets_s,
        'mrl_real': real_mrl_curve,
        'mrl_random_mean': mean_random,
        'mrl_random_ci_lower': ci_lower,
        'mrl_random_ci_upper': ci_upper
    })
    df_results.to_csv(output_csv_path, index=False)
    print(f"Resultados guardados en: {output_csv_path}")

    # -------------------------------------------------------------------------
    # 5. Generar Gráfico Comparativo
    # -------------------------------------------------------------------------
    plot_real_vs_random_offset_summary_ci(
        offsets_s=offsets_s,
        real_mrl=real_mrl_curve,
        random_mean=mean_random,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        output_fig_path=output_fig_path
    )
    print(f"Gráfico guardado en: {output_fig_path}")

    return df_results


def main():
    print("Iniciando lectura de datos...")
    target_fs = None
    user_target = input("Ingrese la frecuencia de muestreo deseada para el subsampleo (por ejemplo 1000), o Enter para usar la original: ").strip()
    if user_target:
        target_fs = int(user_target)
    matriz_eeg, Fs, config = detect_data_source(target_fs=target_fs)

    # Si el usuario solicitó una frecuencia objetivo menor que la original,
    # aplicar filtrado en banda y downsampling por promedio (subsampleo) lo antes posible.
    if target_fs is not None and target_fs < Fs:
        matriz_eeg, Fs = filter_and_downsample_matrix(matriz_eeg, Fs, target_fs, order=2, f1=0.1, f2=150.0)

    n_muestras = matriz_eeg.shape[0]
    n_canales = matriz_eeg.shape[1]

    print("\nInformación del registro")
    print("------------------------")
    print(f"Frecuencia de muestreo: {Fs} Hz")
    print(f"Dimensiones: {matriz_eeg.shape}")
    print(f"Número de canales: {n_canales}")
    print(f"Número de muestras: {n_muestras}")
    print(f"Duración: {n_muestras / Fs:.2f} segundos")
    print(f"Duración: {(n_muestras / Fs) / 60:.2f} minutos")

    if config is not None:
        print("\nConfiguración de canales detectada:")
        for region in config:
            contenido = np.array(config[region])
            print(f"\nRegión: {region}")
            print("Forma:", contenido.shape)
            print(contenido)

    estadisticas = []
    regiones = {
        "CPF": range(0, 10),
        "Nacc": range(10, 16),
        "Amy": range(16, 25),
        "Hyp": range(25, 32)
    }

    for canal in range(n_canales):
        señal = matriz_eeg[:, canal]
        region = ""
        for nombre, canales in regiones.items():
            if canal in canales:
                region = nombre
                break

        promedio = np.mean(señal)
        desviacion = np.std(señal)
        mediana = np.median(señal)
        mad = np.median(np.abs(señal - mediana))
        minimo = np.min(señal)
        maximo = np.max(señal)
        rango = maximo - minimo
        diff = np.diff(señal)
        diff_promedio = np.mean(diff)
        diff_std = np.std(diff)
        diff_mediana = np.median(diff)
        diff_mad = np.median(np.abs(diff - diff_mediana))
        diff_max = np.max(diff)
        diff_min = np.min(diff)
        diff_rango = diff_max - diff_min

        estadisticas.append([
            canal + 1,
            region,
            promedio,
            desviacion,
            mediana,
            mad,
            minimo,
            maximo,
            rango,
            diff_promedio,
            diff_std,
            diff_mediana,
            diff_mad,
            diff_max,
            diff_min,
            diff_rango
        ])

    df = pd.DataFrame(
        estadisticas,
        columns=[
            "Canal",
            "Región",
            "Promedio",
            "Desv_Est",
            "Mediana",
            "MAD",
            "Mínimo",
            "Máximo",
            "Máx-Mín",
            "Diff_promedio",
            "Diff_Desv_Est",
            "Diff_Mediana",
            "Diff_MAD",
            "Diff_Máximo",
            "Diff_Mínimo",
            "Diff_Máx-Mín"
        ]
    )

    print("\nEstadísticos descriptivos")
    print(df)
    df.to_csv("estadisticas_canales.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo guardado como: estadisticas_canales.csv")

    EXCLUDE_29_30 = True
    exclude_set = {28, 29} if EXCLUDE_29_30 else set()

    region_avgs = {}
    for nombre, canales in regiones.items():
        canales_list = [c for c in list(canales) if c not in exclude_set and c < n_canales]
        if len(canales_list) == 0:
            region_avgs[nombre] = np.zeros(n_muestras)
        else:
            region_avgs[nombre] = np.mean(matriz_eeg[:, canales_list], axis=1)

    residual = np.zeros_like(matriz_eeg)
    for ch in range(n_canales):
        region_name = None
        for nombre, canales in regiones.items():
            if ch in canales:
                region_name = nombre
                break
        if region_name is None:
            residual[:, ch] = matriz_eeg[:, ch]
        else:
            residual[:, ch] = matriz_eeg[:, ch] - region_avgs[region_name]

    df_region = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    for nombre, vec in region_avgs.items():
        df_region[f"{nombre}_avg"] = vec
    df_region.to_csv("region_averages.csv", index=False, encoding="utf-8-sig")

    np.save("allChan_residual.npy", residual)
    with h5py.File("allChan_residual.mat", "w") as f_out:
        f_out.create_dataset("allChan_residual", data=residual)
    print("Guardados: region_averages.csv, allChan_residual.npy, allChan_residual.mat")

    segundos_plot = 5
    muestras_plot = int(segundos_plot * Fs)
    muestras_plot = min(muestras_plot, n_muestras)
    tiempo_plot = np.arange(muestras_plot) / Fs
    valid_channels = [c for c in range(n_canales) if c not in exclude_set]
    n_valid = len(valid_channels)

    plt.figure(figsize=(18,12))
    offset = 300
    colores_region = {"CPF":"red", "Nacc":"blue", "Amy":"green", "Hyp":"purple"}

    for idx, ch in enumerate(valid_channels):
        region_name = None
        for nombre, canales in regiones.items():
            if ch in canales:
                region_name = nombre
                break
        color = colores_region.get(region_name, "black")
        plt.plot(tiempo_plot, residual[:muestras_plot, ch] + idx * offset, color=color, linewidth=0.8)

    plt.yticks(np.arange(n_valid) * offset, [f"{ch+1}" for ch in valid_channels])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales (residual)")
    plt.title("Actividad neuronal residual por canal (original - promedio región) - canales excluidos: 29,30")
    plt.grid(True)
    from matplotlib.lines import Line2D
    leyenda = [Line2D([0],[0], color=c, label=r) for r,c in colores_region.items()]
    plt.legend(handles=leyenda, loc="upper right")
    plt.tight_layout()
    plt.savefig("allChan_residual_plot_reduced.png", dpi=300, bbox_inches="tight")
    plt.close()

    residual_reduced = residual[:, valid_channels]
    np.save("allChan_residual_reduced.npy", residual_reduced)
    with h5py.File("allChan_residual_reduced.mat", "w") as f_out_red:
        f_out_red.create_dataset("allChan_residual_reduced", data=residual_reduced)

    # Guardar también la versión "reducida" de la matriz original (sin canales excluidos)
    matriz_eeg_reduced_original = matriz_eeg[:, valid_channels].copy()
    matriz_eeg_residual_reduced = residual_reduced.copy()

    original_channel_indices = valid_channels
    original_to_col = {orig + 1: idx for idx, orig in enumerate(original_channel_indices)}

    # Mantener ambas matrices: original para análisis previos y residual para análisis posteriores
    matriz_eeg_original = matriz_eeg_reduced_original.copy()
    matriz_eeg = matriz_eeg_residual_reduced.copy()
    n_canales = matriz_eeg.shape[1]

    if 1 not in original_to_col or 17 not in original_to_col:
        raise ValueError("Los canales originales 1 o 17 no están disponibles después de la exclusión.")

    canal_1_col = original_to_col[1]
    canal_17_col = original_to_col[17]
    señal_canal_1 = matriz_eeg[:, canal_1_col]
    señal_canal_17 = matriz_eeg[:, canal_17_col]

    filtrado_canal_1 = butterworth_bandpass(señal_canal_1, Fs, 0.0, 30.0, order=4)
    filtrado_canal_17 = butterworth_bandpass(señal_canal_17, Fs, 0.0, 30.0, order=4)

    window_labels, phase_diff_matrix = compute_phase_difference_by_windows(
        señal_canal_1,
        señal_canal_17,
        Fs,
        center_freqs=np.arange(1, 30, dtype=float),
        window_width_hz=2.0,
    )

    np.save("fase_diff_canal_1_17_por_ventanas.npy", phase_diff_matrix)

    phase_summary = pd.DataFrame({
        "Ventana_Hz": window_labels,
        "Promedio_Diferencia_Fase": np.angle(np.mean(np.exp(1j * phase_diff_matrix), axis=1)),
        "MRL": np.abs(np.mean(np.exp(1j * phase_diff_matrix), axis=1)),
    })
    phase_summary.to_csv("phase_difference_summary_by_window.csv", index=False, encoding="utf-8-sig")

    for idx, label in enumerate(window_labels):
        ventana_phase_diff = phase_diff_matrix[idx, :]
        np.save(f"fase_diff_window_{idx + 1}_{label}.npy", ventana_phase_diff)

        df_window = pd.DataFrame({
            "Tiempo_s": np.arange(n_muestras) / Fs,
            "Diferencia_Fase": ventana_phase_diff,
        })
        df_window.to_csv(f"phase_difference_window_{idx + 1}_{label}.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 8))
    plt.imshow(phase_diff_matrix, aspect='auto', cmap='twilight_shifted', origin='lower')
    plt.colorbar(label='Diferencia de fase (rad)')
    plt.xticks(np.arange(0, n_muestras, max(1, n_muestras // 10)), np.arange(0, n_muestras, max(1, n_muestras // 10)) / Fs)
    plt.yticks(np.arange(len(window_labels)), window_labels)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ventana de frecuencia')
    plt.title('Matriz consolidada de diferencia de fase por ventana')
    plt.tight_layout()
    plt.savefig('phase_difference_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matriz de diferencia de fase guardada: fase_diff_canal_1_17_por_ventanas.npy con forma {phase_diff_matrix.shape}")
    print("Resumen por ventana guardado como: phase_difference_summary_by_window.csv")
    print("Archivos individuales por ventana guardados con prefijo: phase_difference_window_...")
    print("Figura guardada como: phase_difference_matrix_heatmap.png")

    window_specs = [
        ("1_6min", 60.0, 360.0),
        ("11_16min", 660.0, 960.0),
    ]

    # ============================================================
    # 11-16 MIN VS 3 PERIODOS ALEATORIOS
    # ============================================================

    save_lagged_mrl_5min_three_random_comparison(
        señal_canal_1,
        señal_canal_17,
        Fs,
        window_name="11_16min",
        real_start_s=660.0,
        real_end_s=960.0,
        pool_start_s=0.0,
        pool_end_s=len(señal_canal_1) / Fs,
        center_freqs=np.arange(1, 30, dtype=float),
        window_width_hz=2.0,
        lags_ms=np.arange(-250, 251, 1, dtype=float),
        random_seed=20260817
    )

    for window_name, start_s, end_s in window_specs:
        df_lagged = compute_lagged_mrl_by_window(
            señal_canal_1,
            señal_canal_17,
            Fs,
            center_freqs=np.arange(1, 30, dtype=float),
            window_width_hz=2.0,
            lags_ms=np.arange(-250, 250, 1),
            start_s=start_s,
            end_s=end_s,
        )
        df_lagged.to_csv(f"mrl_lagged_{window_name}.csv", index=False, encoding="utf-8-sig")

        summary_lagged = df_lagged.groupby("Lag_ms", as_index=False)["MRL"].mean()
        summary_lagged.to_csv(f"mrl_lagged_{window_name}_summary.csv", index=False, encoding="utf-8-sig")

        plot_lagged_mrl_summaries(df_lagged, window_name)

        print(f"Resultados de MRL por desfase guardados para {window_name}: mrl_lagged_{window_name}.csv")

        df_lagged_envolvente = compute_lagged_envelope_correlation_by_window(
            señal_canal_1,
            señal_canal_17,
            Fs,
            center_freqs=np.arange(1, 30, dtype=float),
            window_width_hz=2.0,
            lags_ms=np.arange(-250, 251, 1, dtype=float),
            start_s=start_s,
            end_s=end_s,
        )
        df_lagged_envolvente.to_csv(f"envelope_lagged_{window_name}.csv", index=False, encoding="utf-8-sig")

        summary_lagged_envolvente = df_lagged_envolvente.groupby("Lag_ms", as_index=False)["Correlacion_Envolvente"].mean()
        summary_lagged_envolvente.to_csv(f"envelope_lagged_{window_name}_summary.csv", index=False, encoding="utf-8-sig")

        plot_lagged_envelope_summaries(df_lagged_envolvente, window_name)

        print(f"Resultados de correlación de envolvente por desfase guardados para {window_name}: envelope_lagged_{window_name}.csv")

        save_lagged_mrl_6s_outputs(
            señal_canal_1,
            señal_canal_17,
            Fs,
            window_name,
            start_s,
            end_s,
            center_freqs=np.arange(1, 30, dtype=float),
            window_width_hz=2.0,
            lags_ms=np.arange(-250, 251, 1, dtype=float),
            short_window_s=6.0,
            short_window_step_s=3.0,
        )

        save_lagged_mrl_random_comparison(
            señal_canal_1,
            señal_canal_17,
            Fs,
            window_name,
            start_s,
            end_s,
            center_freqs=np.arange(1, 30, dtype=float),
            window_width_hz=2.0,
            lags_ms=np.arange(-250, 251, 1, dtype=float),
            short_window_s=5.0,
            short_window_step_s=3.0,
            random_seed=20260811 + int(start_s),
        )

    np.save("canal_1_PFC_butterworth_0_30_n4.npy", filtrado_canal_1)
    np.save("canal_17_Amy_butterworth_0_30_n4.npy", filtrado_canal_17)

    band_windows = [(i, i + 2) for i in range(0, 29)]
    df_butter_windows = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_1 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_17 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})

    for lowcut, highcut in band_windows:
        etiqueta = f"{lowcut}_{highcut}Hz"
        filtrado_1 = butterworth_bandpass(señal_canal_1, Fs, float(lowcut), float(highcut), order=4)
        filtrado_17 = butterworth_bandpass(señal_canal_17, Fs, float(lowcut), float(highcut), order=4)
        df_butter_windows[f"Canal_1_PFC_{etiqueta}"] = filtrado_1
        df_butter_windows[f"Canal_17_Amy_{etiqueta}"] = filtrado_17
        df_butter_1[f"Band_{etiqueta}"] = filtrado_1
        df_butter_17[f"Band_{etiqueta}"] = filtrado_17

    df_butter_windows.to_csv("butterworth_order4_canal_1_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_1.to_csv("butterworth_order4_canal_1_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_17.to_csv("butterworth_order4_canal_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")

    print("Guardados: butterworth_order4_canal_1_17_0_30_2hz.csv, butterworth_order4_canal_1_0_30_2hz.csv, butterworth_order4_canal_17_0_30_2hz.csv")
    print("Guardados: versiones individuales .npy para el filtro global 0-30 Hz")

    analytic_canal_1 = hilbert(filtrado_canal_1)
    analytic_canal_17 = hilbert(filtrado_canal_17)
    fase_canal_1 = np.angle(analytic_canal_1)
    fase_canal_17 = np.angle(analytic_canal_17)
    fase_diff_17_1 = np.unwrap(fase_canal_17 - fase_canal_1)

    np.save("fase_canal_1_PFC.npy", fase_canal_1)
    np.save("fase_canal_17_Amy.npy", fase_canal_17)
    np.save("fase_diff_17_1.npy", fase_diff_17_1)

    df_phase_filtered = pd.DataFrame({
        "Tiempo_s": np.arange(n_muestras) / Fs,
        "Fase_Canal_1_PFC": fase_canal_1,
        "Fase_Canal_17_Amy": fase_canal_17,
        "Diferencia_Fase_17_1": fase_diff_17_1
    })
    df_phase_filtered.to_csv("phase_hilbert_canal_1_17_diff.csv", index=False, encoding="utf-8-sig")

    complex_vectors = np.exp(1j * fase_diff_17_1)
    mrl = np.abs(np.mean(complex_vectors))
    print(f"Phase Coherence (MRL): {mrl:.4f}")

    df_mrl = pd.DataFrame({
        "Metric": ["Phase_Coherence_MRL"],
        "Value": [mrl]
    })
    df_mrl.to_csv("phase_coherence_mrl.csv", index=False, encoding="utf-8-sig")

    print("Guardados: phase_hilbert_canal_1_17_diff.csv, phase_coherence_mrl.csv, fase_canal_1_PFC.npy, fase_canal_17_Amy.npy, fase_diff_17_1.npy")

    analytic_signal = hilbert(matriz_eeg, axis=0)
    fase_instantanea = np.angle(analytic_signal)

    np.save("allChan_phase_instantanea.npy", fase_instantanea)
    with h5py.File("allChan_phase_instantanea.mat", "w") as f_phase:
        f_phase.create_dataset("allChan_phase_instantanea", data=fase_instantanea)

    df_phase = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    for idx, origen in enumerate(original_channel_indices):
        canal_label = origen + 1
        df_phase[f"Canal_{canal_label}"] = fase_instantanea[:, idx]
    df_phase.to_csv("allChan_phase_instantanea.csv", index=False, encoding="utf-8-sig")

    def guardar_segmento_csv(nombre_archivo, inicio_s, fin_s):
        inicio = int(inicio_s * Fs)
        fin = int(fin_s * Fs)
        fin = min(fin, n_muestras)
        df_segmento = df_phase.iloc[inicio:fin].copy()
        df_segmento.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

    guardar_segmento_csv("allChan_phase_instantanea_1_6min.csv", 60, 360)
    guardar_segmento_csv("allChan_phase_instantanea_11_16min.csv", 660, 960)

    print("Guardados: allChan_phase_instantanea.npy, allChan_phase_instantanea.mat, allChan_phase_instantanea.csv")
    print("Guardados: allChan_phase_instantanea_1_6min.csv, allChan_phase_instantanea_11_16min.csv")

    n_plot_channels = min(6, n_canales)
    muestras_phase = int(min(2 * Fs, n_muestras))
    tiempo_phase = np.arange(muestras_phase) / Fs
    fase_unwrapped = np.unwrap(fase_instantanea[:muestras_phase, :n_plot_channels], axis=0)

    plt.figure(figsize=(12, 8))
    offset_phase = 10
    for i in range(n_plot_channels):
        plt.plot(tiempo_phase, fase_unwrapped[:, i] + i * offset_phase, label=f"Canal {original_channel_indices[i]+1}")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fase instantánea (rad) + offset")
    plt.title("Fase instantánea de los primeros canales reducidos (Hilbert)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("allChan_phase_instantanea_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Guardado: allChan_phase_instantanea_plot.png")

    canales_a_analizar = sorted(original_to_col.keys())
    print(f"\nSe analizarán {len(canales_a_analizar)} canales (etiquetas originales, excluidos 29 y 30).")

    for canal_label in canales_a_analizar:
        col = original_to_col[canal_label]

        señal_original = matriz_eeg_original[:, col]
        frecuencias_psd_orig, potencia_psd_orig = welch(señal_original, fs=Fs, nperseg=4096)
        plt.figure(figsize=(10,5))
        plt.plot(frecuencias_psd_orig, potencia_psd_orig, color="blue", linewidth=1)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Densidad de potencia")
        plt.title(f"PSD - Canal {canal_label} (original)")
        plt.xlim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"PSD_Canal_{canal_label}_original.png", dpi=300, bbox_inches="tight")
        plt.close()

        frecuencias_spec_orig, tiempos_spec_orig, Sxx_orig = spectrogram(señal_original, fs=Fs, nperseg=1024, noverlap=512)
        plt.figure(figsize=(12,6))
        spec_db_orig = 10 * np.log10(Sxx_orig + 1e-12)
        plt.pcolormesh(tiempos_spec_orig, frecuencias_spec_orig, spec_db_orig, shading="gouraud", cmap="viridis", vmin=-20, vmax=20)
        plt.colorbar(label="Potencia (dB)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title(f"Espectrograma - Canal {canal_label} (original)")
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(f"Espectrograma_Canal_{canal_label}_original.png", dpi=300, bbox_inches="tight")
        plt.close()

        señal_residual = matriz_eeg[:, col]
        frecuencias_psd, potencia_psd = welch(señal_residual, fs=Fs, nperseg=4096)

        plt.figure(figsize=(10,5))
        plt.plot(frecuencias_psd, potencia_psd, color="blue", linewidth=1)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Densidad de potencia")
        plt.title(f"PSD - Canal {canal_label} (residual)")
        plt.xlim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"PSD_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

        frecuencias_spec, tiempos_spec, Sxx = spectrogram(señal_residual, fs=Fs, nperseg=1024, noverlap=512)
        plt.figure(figsize=(12,6))
        spec_db = 10 * np.log10(Sxx + 1e-12)
        plt.pcolormesh(tiempos_spec, frecuencias_spec, spec_db, shading="gouraud", cmap="viridis", vmin=-20, vmax=20)
        plt.colorbar(label="Potencia (dB)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title(f"Espectrograma - Canal {canal_label} (residual)")
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(f"Espectrograma_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10,6))
    regiones_original = {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }

    # PSD promedio por región original
    for nombre, (color, canales_range) in regiones_original.items():
        potencias_psd = []
        for ch_idx in canales_range:
            label = ch_idx + 1
            if label not in original_to_col:
                continue
            col = original_to_col[label]
            señal = matriz_eeg_original[:, col]
            frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
            potencias_psd.append(potencia_psd)
        if len(potencias_psd) == 0:
            continue
        promedio = np.mean(potencias_psd, axis=0)
        plt.plot(frecuencias_psd, promedio, color=color, linewidth=2, label=nombre)

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.title("PSD promedio por región (original)")
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PSD_promedio_regiones_original.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PSD promedio por región residual
    plt.figure(figsize=(10,6))
    for nombre, (color, canales_range) in regiones_original.items():
        potencias_psd = []
        for ch_idx in canales_range:
            label = ch_idx + 1
            if label not in original_to_col:
                continue
            col = original_to_col[label]
            señal = matriz_eeg[:, col]
            frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
            potencias_psd.append(potencia_psd)
        if len(potencias_psd) == 0:
            continue
        promedio = np.mean(potencias_psd, axis=0)
        plt.plot(frecuencias_psd, promedio, color=color, linewidth=2, label=nombre)

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.title("PSD promedio por región (residual)")
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PSD_promedio_regiones_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    analisis_original = []
    analisis_residual = []
    for canal_label in sorted(original_to_col.keys()):
        col = original_to_col[canal_label]

        señal_orig = matriz_eeg_original[:, col]
        frecuencias_psd_orig, potencia_psd_orig = welch(señal_orig, fs=Fs, nperseg=4096)
        indice_orig = np.argmax(potencia_psd_orig)
        frecuencia_pico_orig = frecuencias_psd_orig[indice_orig]
        potencia_pico_orig = potencia_psd_orig[indice_orig]

        señal_residual = matriz_eeg[:, col]
        frecuencias_psd, potencia_psd = welch(señal_residual, fs=Fs, nperseg=4096)
        indice_residual = np.argmax(potencia_psd)
        frecuencia_pico_residual = frecuencias_psd[indice_residual]
        potencia_pico_residual = potencia_psd[indice_residual]

        region = ""
        for nombre, canales_range in regiones_original.items():
            if (canal_label - 1) in canales_range:
                region = nombre
                break

        analisis_original.append([canal_label, region, frecuencia_pico_orig, potencia_pico_orig])
        analisis_residual.append([canal_label, region, frecuencia_pico_residual, potencia_pico_residual])

    df_espectral_original = pd.DataFrame(analisis_original, columns=["Canal", "Región", "Frecuencia_Pico_Hz", "Potencia_Pico"])
    df_espectral_original = df_espectral_original.round(2)
    df_espectral_original.to_csv("Analisis_Espectral_original.csv", index=False, encoding="utf-8-sig")
    print(df_espectral_original)
    print("\nArchivo guardado: Analisis_Espectral_original.csv")

    df_espectral_residual = pd.DataFrame(analisis_residual, columns=["Canal", "Región", "Frecuencia_Pico_Hz", "Potencia_Pico"])
    df_espectral_residual = df_espectral_residual.round(2)
    df_espectral_residual.to_csv("Analisis_Espectral_residual.csv", index=False, encoding="utf-8-sig")
    print(df_espectral_residual)
    print("\nArchivo guardado: Analisis_Espectral_residual.csv")

    print("\nRealizando PCA...")
    X = df.drop(columns=["Canal", "Región"])
    df = df.round(1)
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)
    pca = PCA()
    componentes = pca.fit_transform(X_escalado)
    print(componentes.shape)
    print(componentes[:5, 2])
    print("\nVarianza explicada")
    for i, porcentaje in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {porcentaje*100:.2f}%")

    varianza = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Varianza (%)": pca.explained_variance_ratio_ * 100
    })
    varianza.to_csv("PCA_varianza.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, marker="o")
    plt.xlabel("Componente principal")
    plt.ylabel("Varianza explicada (%)")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_pca = pd.DataFrame({
        "Canal": df["Canal"],
        "Región": df["Región"],
        "PC1": componentes[:,0],
        "PC2": componentes[:,1],
        "PC3": componentes[:,2]
    })

    print("Entrando al gráfico 3D...")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    colores = {"CPF":"red", "Nacc":"blue", "Amy":"green", "Hyp":"purple"}
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        ax.scatter(datos_region["PC1"], datos_region["PC2"], datos_region["PC3"], color=colores[region], label=region, s=80)
        for _, fila in datos_region.iterrows():
            ax.text(fila["PC1"] + 0.03, fila["PC2"] + 0.03, fila["PC3"] + 0.03, str(int(fila["Canal"])), fontsize=9, color=colores[region])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA tridimensional")
    ax.legend()
    plt.tight_layout()
    plt.savefig("PCA_3D.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Gráfico 3D finalizado.")

    plt.figure(figsize=(10,8))
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(datos_region["PC1"], datos_region["PC2"], color=colores[region], s=120, alpha=0.4)
        for _, fila in datos_region.iterrows():
            plt.text(fila["PC1"] + 0.03, fila["PC2"] + 0.03, str(int(fila["Canal"])), fontsize=10, fontweight="bold", color=colores[region], ha="center", va="center")

    loadings = pca.components_.T
    for i, variable in enumerate(X.columns):
        plt.arrow(0, 0, loadings[i,0]*4, loadings[i,1]*4, color="black", alpha=0.7, head_width=0.05)
        plt.text(loadings[i,0]*4.2, loadings[i,1]*4.2, variable, fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot del PCA")
    plt.axhline(0,color="gray",linewidth=0.5)
    plt.axvline(0,color="gray",linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PCA_biplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_pca.to_csv("PCA_canales.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo PCA guardado correctamente.")

    plt.figure(figsize=(8,6))
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(datos_region["PC1"], datos_region["PC2"], color=colores[region], s=120, alpha=0.5, label=region)
        for _, fila in datos_region.iterrows():
            plt.text(fila["PC1"], fila["PC2"], str(int(fila["Canal"])), fontsize=10, fontweight="bold", color=colores[region], ha="center", va="center")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA de los canales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA_scatter_2D.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDetección de artefactos")
    display_channels = sorted(original_to_col.keys())
    for label in display_channels:
        col = original_to_col[label]
        señal = matriz_eeg[:, col]
        diff = np.diff(señal)
        umbral = 5 * np.std(diff)
        artefactos = np.where(np.abs(diff) > umbral)[0]
        print(f"Canal {label}: {len(artefactos)} posibles artefactos")

    segundos = 5
    muestras = int(segundos * Fs)
    muestras = min(muestras, n_muestras)
    tiempo = np.arange(muestras) / Fs
    plt.figure(figsize=(18,12))
    offset = 300
    regiones = {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }
    display_channels = sorted(original_to_col.keys())
    for idx, label in enumerate(display_channels):
        color = "black"
        for nombre_region, (col_color, canales_range) in regiones.items():
            if (label - 1) in canales_range:
                color = col_color
                break
        col = original_to_col[label]
        # Aquí queremos plotear la actividad original reducida (no el residual)
        plt.plot(tiempo, matriz_eeg_reduced_original[:muestras, col] + idx*offset, color=color, linewidth=0.8)

    plt.yticks(np.arange(len(display_channels))*offset, [f"{lbl}" for lbl in display_channels])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales")
    plt.title("Actividad neuronal por regiones cerebrales")
    plt.grid(True)
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0],[0],color="red",label="CPF"),
        Line2D([0],[0],color="blue",label="Nacc"),
        Line2D([0],[0],color="green",label="Amy"),
        Line2D([0],[0],color="purple",label="Hyp")
    ]
    plt.legend(handles=leyenda)
    plt.tight_layout()
    plt.savefig("allChan_activity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Guardado: allChan_activity.png")

# ==============================================================================
# 4. PUNTO DE ENTRADA (INDISPENSABLE PARA MULTIPROCESSING EN WINDOWS)
# ==============================================================================
if __name__ == '__main__':
    save_lagged_mrl_5min_three_random_comparison(
        "allChan_phase_instantanea.mat",
        "resultado_mrl.mat",
        500,
        "Real",
        660,
        960
    )
