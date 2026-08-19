import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats

# ==========================================
# 1. FUNCIONES AUXILIARES
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

def compute_peak_offset(phase1, phase2_full, lags_samples, lags_ms, max_lag_samples):
    """
    Calcula el MRL para cada desfase (lag) y retorna el offset (ms) donde ocurre el MRL máximo.
    """
    N = len(phase1)
    mrls = []
    
    for lag in lags_samples:
        # Desfasar la fase 2 con respecto a la fase 1
        p2_slice = phase2_full[max_lag_samples + lag : max_lag_samples + lag + N]
        phase_diff = p2_slice - phase1
        mrl = np.abs(np.mean(np.exp(1j * phase_diff)))
        mrls.append(mrl)
        
    mrls = np.array(mrls)
    optimal_idx = np.argmax(mrls)
    return lags_ms[optimal_idx]

# ==========================================
# 2. PIPELINE PRINCIPAL DE ANÁLISIS
# ==========================================

def run_phase_offset_analysis(cpf_full, amig_full, fs, win_start_min, win_dur_min=5, n_iterations=50):
    """
    Ejecuta la comparación del offset de máximo MRL para el período real vs. 50 controles al azar.
    """
    # Rangos de frecuencia (1 a 30 Hz, paso de 1 Hz, ancho de banda de 2 Hz)
    center_freqs = np.arange(1, 31, 1)
    
    # Offsets de -250 ms a +250 ms
    max_offset_ms = 250
    max_offset_sec = max_offset_ms / 1000.0
    lags_samples = np.arange(-int(max_offset_sec * fs), int(max_offset_sec * fs) + 1)
    lags_ms = (lags_samples / fs) * 1000.0
    max_lag = int(max_offset_sec * fs)

    # 1. Extraer ventana sincrónica real
    start_idx = int(win_start_min * 60 * fs)
    dur_samples = int(win_dur_min * 60 * fs)
    
    cpf_real = cpf_full[start_idx : start_idx + dur_samples]
    amig_real_padded = amig_full[start_idx - max_lag : start_idx + dur_samples + max_lag]

    # Arrays para almacenar resultados
    real_offsets = []
    random_offsets = np.zeros((n_iterations, len(center_freqs)))

    print(f"Procesando ventana: Minuto {win_start_min} a {win_start_min + win_dur_min}...")

    # 2. Iterar por cada frecuencia
    for f_idx, f_center in enumerate(center_freqs):
        f_low = max(0.1, f_center - 1.0)
        f_high = f_center + 1.0

        # --- CÁLCULO REAL ---
        p1_real = get_phase_signal(cpf_real, f_low, f_high, fs)
        p2_real_padded = get_phase_signal(amig_real_padded, f_low, f_high, fs)
        
        peak_offset_real = compute_peak_offset(p1_real, p2_real_padded, lags_samples, lags_ms, max_lag)
        real_offsets.append(peak_offset_real)

        # --- CÁLCULO CONTROL (AZAR) ---
        # Seleccionar 50 ventanas no coincidentes al azar del resto del registro
        max_random_start = len(cpf_full) - dur_samples - max_lag - 1
        
        for it in range(n_iterations):
            # Posiciones aleatorias garantizadas no sincronizadas
            rand_start_cpf = np.random.randint(0, max_random_start)
            # Asegurar que la ventana de la amígdala esté desfasada/distinta
            rand_start_amig = np.random.randint(0, max_random_start)
            while abs(rand_start_cpf - rand_start_amig) < dur_samples:
                rand_start_amig = np.random.randint(0, max_random_start)

            cpf_rand = cpf_full[rand_start_cpf : rand_start_cpf + dur_samples]
            amig_rand_padded = amig_full[rand_start_amig - max_lag : rand_start_amig + dur_samples + max_lag]

            p1_rand = get_phase_signal(cpf_rand, f_low, f_high, fs)
            p2_rand_padded = get_phase_signal(amig_rand_padded, f_low, f_high, fs)

            random_offsets[it, f_idx] = compute_peak_offset(p1_rand, p2_rand_padded, lags_samples, lags_ms, max_lag)

    return center_freqs, np.array(real_offsets), random_offsets

# ==========================================
# 3. GRAFICADO CON INTERVALO DE CONFIANZA
# ==========================================

def plot_results(center_freqs, real_offsets, random_offsets, title):
    """Grafica la curva real vs. el control al azar con un Intervalo de Confianza del 95%."""
    # Promedio y error estándar (SEM) de las 50 iteraciones al azar
    rand_mean = np.mean(random_offsets, axis=0)
    rand_sem = stats.sem(random_offsets, axis=0)
    
    # Intervalo de Confianza al 95% (1.96 * SEM)
    ci_95 = 1.96 * rand_sem

    plt.figure(figsize=(10, 6))
    
    # Curva Control (Azar)
    plt.plot(center_freqs, rand_mean, label='Control al azar (Promedio 50 iter.)', color='gray', linestyle='--')
    plt.fill_between(center_freqs, rand_mean - ci_95, rand_mean + ci_95, color='gray', alpha=0.3, label='IC 95% (Azar)')

    # Curva Real
    plt.plot(center_freqs, real_offsets, label='Señal Real (Sincrónica)', color='purple', linewidth=2.5, marker='o')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Offset máximo MRL (ms)', fontsize=12)
    plt.ylim(-250, 250)
    plt.axhline(0, color='black', linestyle=':', alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EJECUCIÓN (EJEMPLO DE USO)
# ==========================================

if __name__ == "__main__":
    # --- PARÁMETROS DE CONFIGURACIÓN ---
    FS = 1000  # Frecuencia de muestreo en Hz (Ej: 1000 Hz tras sub-muestreo)
    
    # NOTA: Reemplaza estas líneas cargando tus datos reales de NumPy
    # cpf_signal_channel_1 = np.load("cpf_ch1.npy")
    # amigdala_signal_channel_1 = np.load("amig_ch1.npy")
    
    # Simulando señales de prueba (Ejemplo de 20 minutos de registro)
    duration_sec = 20 * 60
    t = np.linspace(0, duration_sec, int(FS * duration_sec))
    cpf_signal_channel_1 = np.sin(2 * np.pi * 8 * t) + np.random.normal(0, 0.5, len(t))
    amigdala_signal_channel_1 = np.sin(2 * np.pi * 8 * (t - 0.05)) + np.random.normal(0, 0.5, len(t))

    # --- BLOQUE 1: Minuto 1 al 6 (Morada) ---
    freqs1, real_off1, rand_off1 = run_phase_offset_analysis(
        cpf_full=cpf_signal_channel_1,
        amig_full=amigdala_signal_channel_1,
        fs=FS,
        win_start_min=1,
        win_dur_min=5,
        n_iterations=50
    )
    plot_results(freqs1, real_off1, rand_off1, "Minuto 1 a 6 (Período Morada) - Real vs. Azar")

    # --- BLOQUE 2: Minuto 11 al 16 (Laberinto) ---
    freqs2, real_off2, rand_off2 = run_phase_offset_analysis(
        cpf_full=cpf_signal_channel_1,
        amig_full=amigdala_signal_channel_1,
        fs=FS,
        win_start_min=11,
        win_dur_min=5,
        n_iterations=50
    )
    plot_results(freqs2, real_off2, rand_off2, "Minuto 11 a 16 (Período Laberinto) - Real vs. Azar")
