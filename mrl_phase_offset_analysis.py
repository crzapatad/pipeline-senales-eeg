import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar bloqueos
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
from typing import Tuple, Dict, List
import warnings
import sys
warnings.filterwarnings('ignore')

def log_print(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================

def cargar_datos_lfp(ruta_cpf: str, ruta_amig: str, fs: float = 1000.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Carga las señales LFP de CPF y Amígdala desde archivos .npy.
    
    Args:
        ruta_cpf: Ruta al archivo .npy del canal CPF
        ruta_amig: Ruta al archivo .npy del canal Amígdala
        fs: Frecuencia de muestreo en Hz
        
    Returns:
        tuple: (señal_cpf, señal_amig, fs)
    """
    log_print("Cargando datos...")
    señal_cpf = np.load(ruta_cpf)
    log_print("CPF cargado")
    señal_amig = np.load(ruta_amig)
    log_print("Amígdala cargada")
    
    # Verificar que las señales tengan la misma longitud
    if len(señal_cpf) != len(señal_amig):
        min_len = min(len(señal_cpf), len(señal_amig))
        señal_cpf = señal_cpf[:min_len]
        señal_amig = señal_amig[:min_len]
        log_print(f"ADVERTENCIA: Señales recortadas a {min_len} muestras para igualar longitudes")
    
    log_print(f"Datos cargados: CPF={len(señal_cpf)} muestras, Amígdala={len(señal_amig)} muestras, fs={fs} Hz")
    return señal_cpf, señal_amig, fs

# ==========================================
# 2. FUNCIONES DE PROCESAMIENTO DE SEÑAL
# ==========================================

def filtro_pasabanda(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    """
    Aplica un filtro Butterworth pasabanda de orden 3.
    
    Args:
        data: Señal a filtrar
        lowcut: Frecuencia de corte inferior (Hz)
        highcut: Frecuencia de corte superior (Hz)
        fs: Frecuencia de muestreo (Hz)
        order: Orden del filtro (default=3)
        
    Returns:
        np.ndarray: Señal filtrada
    """
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extraer_fase_instantanea(signal: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """
    Filtra la señal en la banda deseada y extrae la fase instantánea mediante Hilbert.
    
    Args:
        signal: Señal original
        lowcut: Frecuencia de corte inferior (Hz)
        highcut: Frecuencia de corte superior (Hz)
        fs: Frecuencia de muestreo (Hz)
        
    Returns:
        np.ndarray: Fase instantánea en radianes
    """
    filtrada = filtro_pasabanda(signal, lowcut, highcut, fs)
    analitica = hilbert(filtrada)
    return np.angle(analitica)

def calcular_mrl_con_lags(fase1: np.ndarray, fase2_full: np.ndarray, 
                          lags_samples: np.ndarray, max_lag_samples: int) -> np.ndarray:
    """
    Calcula el MRL para cada desfase (lag) entre dos fases.
    
    Args:
        fase1: Fase del canal 1 (ventana de análisis)
        fase2_full: Fase del canal 2 (ventana con padding para lags)
        lags_samples: Array de lags en muestras
        max_lag_samples: Máximo lag en muestras (para padding)
        
    Returns:
        np.ndarray: Array de MRL para cada lag
    """
    N = len(fase1)
    mrls = []
    
    for lag in lags_samples:
        # Extraer segmentos desfasados
        p2_slice = fase2_full[max_lag_samples + lag : max_lag_samples + lag + N]
        
        # Verificar que los tamaños coincidan
        if len(p2_slice) != N:
            # Si no coinciden, ajustar el tamaño de fase1
            min_len = min(len(p2_slice), N)
            p2_slice = p2_slice[:min_len]
            fase1_adjusted = fase1[:min_len]
            fase_diff = p2_slice - fase1_adjusted
        else:
            fase_diff = p2_slice - fase1
            
        mrl = np.abs(np.mean(np.exp(1j * fase_diff)))
        mrls.append(mrl)
    
    return np.array(mrls)

def obtener_offset_mrl_maximo(mrls: np.ndarray, lags_ms: np.ndarray) -> float:
    """
    Obtiene el offset (ms) donde ocurre el MRL máximo.
    
    Args:
        mrls: Array de valores MRL
        lags_ms: Array de lags correspondientes en ms
        
    Returns:
        float: Offset en ms donde se produce el MRL máximo
    """
    idx_maximo = np.argmax(mrls)
    return lags_ms[idx_maximo]

# ==========================================
# 3. ANÁLISIS DE VENTANAS
# ==========================================

def extraer_ventana_sincronica(señal: np.ndarray, start_min: float, dur_min: float, 
                                fs: float, max_lag_samples: int) -> np.ndarray:
    """
    Extrae una ventana sincronizada con padding para lags.
    
    Args:
        señal: Señal completa
        start_min: Tiempo de inicio en minutos
        dur_min: Duración en minutos
        fs: Frecuencia de muestreo
        max_lag_samples: Padding máximo para lags
        
    Returns:
        np.ndarray: Ventana con padding
    """
    start_idx = int(start_min * 60 * fs)
    dur_samples = int(dur_min * 60 * fs)
    
    # Extraer ventana con padding
    señal_padded = señal[start_idx - max_lag_samples : start_idx + dur_samples + max_lag_samples]
    return señal_padded

def extraer_ventana_sin_padding(señal: np.ndarray, start_min: float, dur_min: float, 
                                fs: float) -> np.ndarray:
    """
    Extrae una ventana sin padding para análisis básico.
    
    Args:
        señal: Señal completa
        start_min: Tiempo de inicio en minutos
        dur_min: Duración en minutos
        fs: Frecuencia de muestreo
        
    Returns:
        np.ndarray: Ventana extraída
    """
    start_idx = int(start_min * 60 * fs)
    dur_samples = int(dur_min * 60 * fs)
    return señal[start_idx : start_idx + dur_samples]

def generar_ventanas_aleatorias_no_coincidentes(señal_cpf: np.ndarray, señal_amig: np.ndarray,
                                                dur_min: float, fs: float, 
                                                ventana_real_start: float, ventana_real_end: float,
                                                n_ventanas: int, semilla: int = 42) -> List[Tuple[float, float]]:
    """
    Genera ventanas aleatorias no coincidentes con la ventana real ni entre sí.
    
    Args:
        señal_cpf: Señal CPF completa
        señal_amig: Señal Amígdala completa
        dur_min: Duración de cada ventana en minutos
        fs: Frecuencia de muestreo
        ventana_real_start: Inicio de ventana real (min)
        ventana_real_end: Fin de ventana real (min)
        n_ventanas: Número de ventanas aleatorias a generar
        semilla: Semilla para reproducibilidad
        
    Returns:
        List[Tuple]: Lista de tuplas (start_min, end_min) para cada ventana aleatoria
    """
    np.random.seed(semilla)
    
    total_muestras = len(señal_cpf)
    dur_samples = int(dur_min * 60 * fs)
    dur_sec = dur_min * 60
    
    ventanas = []
    max_intentos = 10000
    intentos = 0
    
    while len(ventanas) < n_ventanas and intentos < max_intentos:
        intentos += 1
        
        # Generar posición aleatoria (simplificado - misma posición para ambas señales)
        max_start = (total_muestras - dur_samples) / (fs * 60)
        start_cpf = np.random.uniform(0, max_start)
        end_cpf = start_cpf + dur_min
        
        # Verificar que no se solape con ventana real
        if not (start_cpf >= ventana_real_end or end_cpf <= ventana_real_start):
            continue
        
        # Verificar que no se solape con ventanas existentes
        solapamiento = False
        for existing_start, existing_end in ventanas:
            if not (start_cpf >= existing_end or end_cpf <= existing_start):
                solapamiento = True
                break
        
        if not solapamiento:
            ventanas.append((start_cpf, end_cpf))
    
    if len(ventanas) < n_ventanas:
        print(f"ADVERTENCIA: Solo se pudieron generar {len(ventanas)} de {n_ventanas} ventanas aleatorias")
    
    return ventanas

# ==========================================
# 4. ANÁLISIS PRINCIPAL DE MRL
# ==========================================

def analizar_offset_mrl_periodo(señal_cpf: np.ndarray, señal_amig: np.ndarray, fs: float,
                                start_min: float, dur_min: float,
                                freq_range: Tuple[int, int] = (1, 30),
                                lag_range_ms: Tuple[int, int] = (-250, 250),
                                n_iteraciones_azar: int = 50,
                                n_controles_individuales: int = 3,
                                semilla: int = 42) -> Dict:
    """
    Analiza el offset de MRL máximo para un período de tiempo específico.
    
    Args:
        señal_cpf: Señal CPF completa
        señal_amig: Señal Amígdala completa
        fs: Frecuencia de muestreo
        start_min: Inicio del período (minutos)
        dur_min: Duración del período (minutos)
        freq_range: Rango de frecuencias (low, high) Hz
        lag_range_ms: Rango de lags (min, max) ms
        n_iteraciones_azar: Número de iteraciones para control global
        n_controles_individuales: Número de controles individuales específicos
        semilla: Semilla para reproducibilidad
        
    Returns:
        Dict: Diccionario con resultados del análisis
    """
    # Configuración de parámetros
    freqs = np.arange(freq_range[0], freq_range[1] + 1, 1)
    max_lag_ms = max(abs(lag_range_ms[0]), abs(lag_range_ms[1]))
    max_lag_sec = max_lag_ms / 1000.0
    max_lag_samples = int(max_lag_sec * fs)
    
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = (lags_samples / fs) * 1000.0
    
    log_print(f"\n=== ANALIZANDO PERÍODO: Min {start_min} a {start_min + dur_min} ===")
    log_print(f"Configuración: {len(freqs)} frecuencias, {len(lags_samples)} lags, {n_iteraciones_azar} iteraciones")
    
    # 1. Extraer ventana real sincrónica
    log_print("  Extrayendo ventanas reales...")
    ventana_real_cpf = extraer_ventana_sincronica(señal_cpf, start_min, dur_min, fs, max_lag_samples)
    ventana_real_amig = extraer_ventana_sincronica(señal_amig, start_min, dur_min, fs, max_lag_samples)
    ventana_real_cpf_sin_pad = extraer_ventana_sin_padding(señal_cpf, start_min, dur_min, fs)
    log_print(f"  Ventanas extraídas: CPF={ventana_real_cpf.shape}, Amígdala={ventana_real_amig.shape}")
    
    # Arrays para almacenar resultados
    offsets_reales = []
    offsets_azar = np.zeros((n_iteraciones_azar, len(freqs)))
    offsets_controles_individuales = []
    
    # 2. Generar ventanas aleatorias para controles
    log_print("  Generando ventanas aleatorias...")
    ventana_real_end = start_min + dur_min
    ventanas_azar = generar_ventanas_aleatorias_no_coincidentes(
        señal_cpf, señal_amig, dur_min, fs, start_min, ventana_real_end,
        max(n_iteraciones_azar, n_controles_individuales), semilla
    )
    log_print(f"  Ventanas aleatorias generadas: {len(ventanas_azar)}")
    
    # 3. Calcular para cada frecuencia
    for f_idx, freq_center in enumerate(freqs):
        log_print(f"  Procesando frecuencia {freq_center} Hz ({f_idx+1}/{len(freqs)})...")
        f_low = max(0.1, freq_center - 1.0)
        f_high = freq_center + 1.0
        
        # --- CÁLCULO REAL ---
        fase_cpf_real = extraer_fase_instantanea(ventana_real_cpf_sin_pad, f_low, f_high, fs)
        fase_amig_real_padded = extraer_fase_instantanea(ventana_real_amig, f_low, f_high, fs)
        
        mrls_real = calcular_mrl_con_lags(fase_cpf_real, fase_amig_real_padded, lags_samples, max_lag_samples)
        offset_real = obtener_offset_mrl_maximo(mrls_real, lags_ms)
        offsets_reales.append(offset_real)
        
        # --- CÁLCULO ITERACIONES AL AZAR (CONTROL GLOBAL) ---
        # Desactivado temporalmente para pruebas
        # for it in range(min(n_iteraciones_azar, len(ventanas_azar))):
        #     start_rand, end_rand = ventanas_azar[it]
        #     
        #     # Ventanas aleatorias desincronizadas (usar mismo tiempo base pero diferentes offset)
        #     cpf_rand = extraer_ventana_sincronica(señal_cpf, start_rand, dur_min, fs, max_lag_samples)
        #     amig_rand = extraer_ventana_sincronica(señal_amig, start_rand, dur_min, fs, max_lag_samples)
        #     
        #     cpf_rand_sin_pad = extraer_ventana_sin_padding(señal_cpf, start_rand, dur_min, fs)
        #     
        #     fase_cpf_rand = extraer_fase_instantanea(cpf_rand_sin_pad, f_low, f_high, fs)
        #     fase_amig_rand_padded = extraer_fase_instantanea(amig_rand, f_low, f_high, fs)
        #     
        #     mrls_rand = calcular_mrl_con_lags(fase_cpf_rand, fase_amig_rand_padded, lags_samples, max_lag_samples)
        #     offset_rand = obtener_offset_mrl_maximo(mrls_rand, lags_ms)
        #     offsets_azar[it, f_idx] = offset_rand
        
        # Rellenar con ceros para mantener estructura
        for it in range(n_iteraciones_azar):
            offsets_azar[it, f_idx] = 0.0
    
    # --- CÁLCULO CONTROLES INDIVIDUALES ESPECÍFICOS ---
    # Desactivado temporalmente para pruebas
    # Usar las primeras n_controles_individuales ventanas como controles individuales
    # for i in range(min(n_controles_individuales, len(ventanas_azar))):
    #     offsets_control_indiv = []
    #     start_rand, end_rand = ventanas_azar[i]
    #     
    #     cpf_rand = extraer_ventana_sincronica(señal_cpf, start_rand, dur_min, fs, max_lag_samples)
    #     amig_rand = extraer_ventana_sincronica(señal_amig, start_rand, dur_min, fs, max_lag_samples)
    #     cpf_rand_sin_pad = extraer_ventana_sin_padding(señal_cpf, start_rand, dur_min, fs)
    #     
    #     for f_idx, freq_center in enumerate(freqs):
    #         f_low = max(0.1, freq_center - 1.0)
    #         f_high = freq_center + 1.0
    #         
    #         fase_cpf_rand = extraer_fase_instantanea(cpf_rand_sin_pad, f_low, f_high, fs)
    #         fase_amig_rand_padded = extraer_fase_instantanea(amig_rand, f_low, f_high, fs)
    #         
    #         mrls_rand = calcular_mrl_con_lags(fase_cpf_rand, fase_amig_rand_padded, lags_samples, max_lag_samples)
    #         offset_rand = obtener_offset_mrl_maximo(mrls_rand, lags_ms)
    #         offsets_control_indiv.append(offset_rand)
    #     
    #     offsets_controles_individuales.append(offsets_control_indiv)
    
    log_print("  Controles individuales desactivados temporalmente")
    
    return {
        'frecuencias': freqs,
        'offsets_reales': np.array(offsets_reales),
        'offsets_azar': offsets_azar,
        'offsets_controles_individuales': offsets_controles_individuales,
        'lags_ms': lags_ms
    }

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================

def graficar_resultados_mrl(resultados: Dict, titulo_periodo: str, guardar_fig: bool = False,
                            ruta_guardado: str = None):
    """
    Grafica los resultados del análisis de MRL para un período.
    
    Args:
        resultados: Diccionario con resultados del análisis
        titulo_periodo: Título descriptivo del período
        guardar_fig: Si True, guarda la figura
        ruta_guardado: Ruta para guardar la figura
    """
    freqs = resultados['frecuencias']
    offsets_reales = resultados['offsets_reales']
    offsets_azar = resultados['offsets_azar']
    offsets_controles_individuales = resultados['offsets_controles_individuales']
    
    # Calcular estadísticas del control global
    media_azar = np.mean(offsets_azar, axis=0)
    sem_azar = stats.sem(offsets_azar, axis=0)
    ic_95 = 1.96 * sem_azar
    
    # Crear figura
    plt.figure(figsize=(12, 7))
    
    # 1. Curva Real Sincrónica (destacada)
    plt.plot(freqs, offsets_reales, label='Real Sincrónico', color='purple', 
             linewidth=2.5, marker='o', markersize=4, zorder=5)
    
    # 2. Promedio 50 iteraciones al azar con IC 95%
    plt.plot(freqs, media_azar, label='Promedio 50 iteraciones azar', color='gray', 
             linestyle='--', linewidth=2, alpha=0.8)
    plt.fill_between(freqs, media_azar - ic_95, media_azar + ic_95, 
                     color='gray', alpha=0.3, label='IC 95% (Azar)')
    
    # 3. 3 Curvas de control individuales
    colores_individuales = ['orange', 'green', 'red']
    estilos_individuales = [':', '-.', '--']
    
    for i, offsets_indiv in enumerate(offsets_controles_individuales):
        color = colores_individuales[i % len(colores_individuales)]
        estilo = estilos_individuales[i % len(estilos_individuales)]
        plt.plot(freqs, offsets_indiv, label=f'Control Individual {i+1}', 
                 color=color, linestyle=estilo, linewidth=1.2, alpha=0.6)
    
    # Formato del gráfico
    plt.title(f'Análisis de Offset MRL Máximo - {titulo_periodo}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Offset MRL Máximo (ms)', fontsize=12)
    plt.xlim(freqs[0] - 0.5, freqs[-1] + 0.5)
    plt.ylim(-250, 250)
    
    # Línea de referencia en y=0
    plt.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Offset 0 ms')
    
    # Grid y leyenda
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if guardar_fig and ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {ruta_guardado}")
    
    plt.show()

# ==========================================
# 6. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==========================================

def ejecutar_analisis_completo(ruta_cpf: str, ruta_amig: str, fs: float = 1000.0,
                                freq_range: Tuple[int, int] = (1, 30),
                                lag_range_ms: Tuple[int, int] = (-250, 250),
                                n_iteraciones_azar: int = 50,
                                n_controles_individuales: int = 3):
    """
    Ejecuta el análisis completo para ambos períodos (Morada y Laberinto).
    
    Args:
        ruta_cpf: Ruta al archivo .npy del canal CPF
        ruta_amig: Ruta al archivo .npy del canal Amígdala
        fs: Frecuencia de muestreo (Hz)
        freq_range: Rango de frecuencias (low, high) Hz
        lag_range_ms: Rango de lags (min, max) ms
        n_iteraciones_azar: Número de iteraciones para control global
        n_controles_individuales: Número de controles individuales
    """
    # Cargar datos
    señal_cpf, señal_amig, fs = cargar_datos_lfp(ruta_cpf, ruta_amig, fs)
    
    # Configuración de períodos
    periodos = [
        ("Morada", 1, 6),
        ("Laberinto", 11, 16)
    ]
    
    resultados_completos = {}
    
    # Analizar cada período
    for nombre, start_min, dur_min in periodos:
        log_print(f"\n{'='*60}")
        log_print(f"INICIANDO ANÁLISIS: {nombre} (Min {start_min}-{start_min+dur_min})")
        log_print(f"{'='*60}")
        
        resultados = analizar_offset_mrl_periodo(
            señal_cpf, señal_amig, fs,
            start_min, dur_min,
            freq_range, lag_range_ms,
            n_iteraciones_azar, n_controles_individuales,
            semilla=42 + int(start_min)
        )
        
        resultados_completos[nombre] = resultados
        
        # Graficar resultados (desactivado temporalmente para pruebas)
        # titulo = f"Min {start_min} a {start_min + dur_min} ({nombre})"
        # graficar_resultados_mrl(resultados, titulo, guardar_fig=False)
        log_print(f"  Análisis de {nombre} completado (gráficos desactivados temporalmente)")
    
    return resultados_completos

# ==========================================
# 7. BLOQUE DE EJECUCIÓN (CONFIGURACIÓN)
# ==========================================

if __name__ == "__main__":
    # ========================================
    # CONFIGURACIÓN DE RUTAS Y PARÁMETROS
    # ========================================
    
    # Rutas a los archivos .npy (MODIFICAR SEGÚN TU ESTRUCTURA DE ARCHIVOS)
    RUTA_CPF = "canal_1_PFC_butterworth_0_30_n4.npy"      # Canal 1 (CPF)
    RUTA_AMIG = "canal_17_Amy_butterworth_0_30_n4.npy"    # Canal 17 (Amígdala)
    
    # Frecuencia de muestreo (Hz)
    FS = 1000.0  # Modificar según tu frecuencia de muestreo real
    
    # Parámetros del análisis (reducidos para pruebas - cambiar a valores originales para análisis completo)
    RANGO_FRECUENCIAS = (1, 5)            # 1 Hz a 5 Hz (muy reducido para pruebas ultra rápidas)
    RANGO_LAGS_MS = (-50, 50)             # -50 ms a +50 ms (muy reducido para pruebas ultra rápidas)
    N_ITERACIONES_AZAR = 2                # Iteraciones para control global (muy reducido para pruebas ultra rápidas)
    N_CONTROLES_INDIVIDUALES = 1          # Controles individuales específicos (muy reducido para pruebas ultra rápidas)
    
    # Para análisis completo, usar estos valores:
    # RANGO_FRECUENCIAS = (1, 30)           # 1 Hz a 30 Hz
    # RANGO_LAGS_MS = (-250, 250)           # -250 ms a +250 ms
    # N_ITERACIONES_AZAR = 50               # Iteraciones para control global
    # N_CONTROLES_INDIVIDUALES = 3          # Controles individuales específicos
    
    # ========================================
    # EJECUCIÓN DEL ANÁLISIS
    # ========================================
    
    log_print("=" * 60)
    log_print("ANÁLISIS DE ACOPLAMIENTO DE FASE MRL")
    log_print("CPF vs Amígdala - Offset de MRL Máximo")
    log_print("=" * 60)
    
    try:
        resultados = ejecutar_analisis_completo(
            ruta_cpf=RUTA_CPF,
            ruta_amig=RUTA_AMIG,
            fs=FS,
            freq_range=RANGO_FRECUENCIAS,
            lag_range_ms=RANGO_LAGS_MS,
            n_iteraciones_azar=N_ITERACIONES_AZAR,
            n_controles_individuales=N_CONTROLES_INDIVIDUALES
        )
        
        log_print("\n" + "=" * 60)
        log_print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        log_print("=" * 60)
        
    except FileNotFoundError as e:
        log_print(f"\nERROR: No se encontraron los archivos de datos.")
        log_print(f"Por favor, verifica las rutas en la sección de configuración.")
        log_print(f"Error específico: {e}")
        
    except Exception as e:
        log_print(f"\nERROR durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

