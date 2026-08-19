import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from typing import Tuple, Dict, List
import warnings
import sys
warnings.filterwarnings('ignore')

def log_print(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# ==========================================
# 1. FUNCIONES SIMPLIFICADAS
# ==========================================

def cargar_datos_lfp(ruta_cpf: str, ruta_amig: str, fs: float = 1000.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Carga las señales LFP de CPF y Amígdala desde archivos .npy."""
    log_print("Cargando datos...")
    señal_cpf = np.load(ruta_cpf)
    log_print("CPF cargado")
    señal_amig = np.load(ruta_amig)
    log_print("Amígdala cargada")
    
    if len(señal_cpf) != len(señal_amig):
        min_len = min(len(señal_cpf), len(señal_amig))
        señal_cpf = señal_cpf[:min_len]
        señal_amig = señal_amig[:min_len]
        log_print(f"ADVERTENCIA: Señales recortadas a {min_len} muestras para igualar longitudes")
    
    log_print(f"Datos cargados: CPF={len(señal_cpf)} muestras, Amígdala={len(señal_amig)} muestras, fs={fs} Hz")
    return señal_cpf, señal_amig, fs

def filtro_pasabanda(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    """Aplica un filtro Butterworth pasabanda de orden 3."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extraer_fase_instantanea(signal: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Filtra la señal en la banda deseada y extrae la fase instantánea mediante Hilbert."""
    filtrada = filtro_pasabanda(signal, lowcut, highcut, fs)
    analitica = hilbert(filtrada)
    return np.angle(analitica)

def calcular_mrl_con_lags(fase1: np.ndarray, fase2: np.ndarray, lags_samples: np.ndarray) -> np.ndarray:
    """Calcula el MRL para cada desfase (lag) entre dos fases."""
    N = len(fase1)
    mrls = []
    
    for lag in lags_samples:
        if lag < 0:
            s1 = fase1[-lag:]
            s2 = fase2[:N + lag]
        elif lag > 0:
            s1 = fase1[:N - lag]
            s2 = fase2[lag:]
        else:
            s1 = fase1
            s2 = fase2
            
        diferencia_fase = np.exp(1j * (s1 - s2))
        mrl = np.abs(np.mean(diferencia_fase))
        mrls.append(mrl)
    
    return np.array(mrls)

def obtener_offset_mrl_maximo(mrls: np.ndarray, lags_ms: np.ndarray) -> float:
    """Obtiene el offset (ms) donde ocurre el MRL máximo."""
    idx_maximo = np.argmax(mrls)
    return lags_ms[idx_maximo]

# ==========================================
# 2. ANÁLISIS SIMPLIFICADO
# ==========================================

def analizar_periodo_simplificado(señal_cpf: np.ndarray, señal_amig: np.ndarray, fs: float,
                                start_min: float, dur_min: float,
                                freq_range: Tuple[int, int] = (1, 5),
                                lag_range_ms: Tuple[int, int] = (-100, 100)) -> Dict:
    """Analiza el offset de MRL máximo para un período de tiempo específico."""
    
    freqs = np.arange(freq_range[0], freq_range[1] + 1, 1)
    max_lag_ms = max(abs(lag_range_ms[0]), abs(lag_range_ms[1]))
    max_lag_sec = max_lag_ms / 1000.0
    max_lag_samples = int(max_lag_sec * fs)
    
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = (lags_samples / fs) * 1000.0
    
    log_print(f"\n=== ANALIZANDO PERÍODO: Min {start_min} a {start_min + dur_min} ===")
    log_print(f"Configuración: {len(freqs)} frecuencias, {len(lags_samples)} lags")
    
    # Extraer ventana
    start_idx = int(start_min * 60 * fs)
    dur_samples = int(dur_min * 60 * fs)
    
    cpf_window = señal_cpf[start_idx:start_idx + dur_samples]
    amig_window = señal_amig[start_idx:start_idx + dur_samples]
    
    log_print(f"Ventana extraída: CPF={cpf_window.shape}, Amígdala={amig_window.shape}")
    
    offsets_reales = []
    
    # Calcular para cada frecuencia
    for f_idx, freq_center in enumerate(freqs):
        log_print(f"  Procesando frecuencia {freq_center} Hz ({f_idx+1}/{len(freqs)})...")
        f_low = max(0.1, freq_center - 1.0)
        f_high = freq_center + 1.0
        
        fase_cpf = extraer_fase_instantanea(cpf_window, f_low, f_high, fs)
        fase_amig = extraer_fase_instantanea(amig_window, f_low, f_high, fs)
        
        mrls = calcular_mrl_con_lags(fase_cpf, fase_amig, lags_samples)
        offset = obtener_offset_mrl_maximo(mrls, lags_ms)
        offsets_reales.append(offset)
        log_print(f"    Offset máximo MRL: {offset:.2f} ms")
    
    return {
        'frecuencias': freqs,
        'offsets_reales': np.array(offsets_reales),
        'lags_ms': lags_ms
    }

# ==========================================
# 3. EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    log_print("=" * 60)
    log_print("ANÁLISIS DE ACOPLAMIENTO DE FASE MRL (SIMPLIFICADO)")
    log_print("CPF vs Amígdala - Offset de MRL Máximo")
    log_print("=" * 60)
    
    # Configuración
    RUTA_CPF = "canal_1_PFC_butterworth_0_30_n4.npy"
    RUTA_AMIG = "canal_17_Amy_butterworth_0_30_n4.npy"
    FS = 1000.0
    RANGO_FRECUENCIAS = (1, 5)
    RANGO_LAGS_MS = (-100, 100)
    
    try:
        # Cargar datos
        señal_cpf, señal_amig, fs = cargar_datos_lfp(RUTA_CPF, RUTA_AMIG, FS)
        
        # Analizar períodos
        periodos = [
            ("Morada", 1, 6),
            ("Laberinto", 11, 16)
        ]
        
        resultados_completos = {}
        
        for nombre, start_min, dur_min in periodos:
            log_print(f"\n{'='*60}")
            log_print(f"INICIANDO ANÁLISIS: {nombre} (Min {start_min}-{start_min+dur_min})")
            log_print(f"{'='*60}")
            
            resultados = analizar_periodo_simplificado(
                señal_cpf, señal_amig, fs,
                start_min, dur_min,
                RANGO_FRECUENCIAS, RANGO_LAGS_MS
            )
            
            resultados_completos[nombre] = resultados
            log_print(f"  Análisis de {nombre} completado")
        
        log_print("\n" + "=" * 60)
        log_print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        log_print("=" * 60)
        
        # Mostrar resumen
        for nombre, resultados in resultados_completos.items():
            log_print(f"\n{nombre}:")
            log_print(f"  Frecuencias: {resultados['frecuencias']}")
            log_print(f"  Offsets: {resultados['offsets_reales']}")
        
    except Exception as e:
        log_print(f"\nERROR durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
