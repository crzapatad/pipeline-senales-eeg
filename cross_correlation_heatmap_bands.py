import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import pandas as pd

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Aplica filtro Butterworth pasabanda."""
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999999)
    if low <= 1e-6:
        b, a = butter(order, high, btype='low')
    else:
        b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_cross_correlation_by_band(signal1, signal2, fs, freq_bands, max_lag_ms=250):
    """
    Calcula correlación cruzada para diferentes bandas de frecuencia.
    
    Parámetros:
    - signal1, signal2: señales a analizar
    - fs: frecuencia de muestreo
    - freq_bands: lista de tuplas (low, high) para las bandas de frecuencia
    - max_lag_ms: máximo desfase en milisegundos
    
    Retorna:
    - correlation_matrix: matriz de correlación cruzada (frecuencias x lags)
    - lags_ms: array de lags en milisegundos
    - freq_labels: etiquetas de las bandas de frecuencia
    """
    max_lag_samples = int(max_lag_ms / 1000.0 * fs)
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = lags_samples / fs * 1000.0
    
    correlation_matrix = np.zeros((len(freq_bands), len(lags_ms)))
    freq_labels = []
    
    for i, (lowcut, highcut) in enumerate(freq_bands):
        # Filtrar ambas señales en la banda de frecuencia
        filtered1 = bandpass_filter(signal1, lowcut, highcut, fs)
        filtered2 = bandpass_filter(signal2, lowcut, highcut, fs)
        
        # Normalizar señales (z-score)
        filtered1 = (filtered1 - np.mean(filtered1)) / (np.std(filtered1) + 1e-10)
        filtered2 = (filtered2 - np.mean(filtered2)) / (np.std(filtered2) + 1e-10)
        
        # Limpiar NaN/Inf
        filtered1 = np.nan_to_num(filtered1, nan=0.0, posinf=0.0, neginf=0.0)
        filtered2 = np.nan_to_num(filtered2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Usar ventana más pequeña para velocidad
        window_size = min(len(filtered1), 100000)
        filtered1_win = filtered1[:window_size]
        filtered2_win = filtered2[:window_size]
        
        # Calcular correlación cruzada
        cross_corr = correlate(filtered1_win, filtered2_win, mode='same', method='auto')
        
        # Extraer rango de lags
        center_idx = len(cross_corr) // 2
        start_idx = center_idx - max_lag_samples
        end_idx = center_idx + max_lag_samples + 1
        
        cross_corr_limited = cross_corr[start_idx:end_idx]
        
        # Normalizar
        n = len(filtered1_win)
        lags_range = np.arange(-max_lag_samples, max_lag_samples + 1)
        normalization = n - np.abs(lags_range)
        normalization[normalization == 0] = 1
        
        cross_corr_normalized = cross_corr_limited / normalization
        cross_corr_normalized = np.nan_to_num(cross_corr_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        correlation_matrix[i, :] = cross_corr_normalized
        
        freq_label = f"Banda {lowcut:g}-{highcut:g} Hz"
        freq_labels.append(freq_label)
        print(f"Procesada banda {freq_label}")
    
    return correlation_matrix, lags_ms, freq_labels

def plot_heatmap(correlation_matrix, lags_ms, freq_labels, title="Correlación Cruzada por Bandas de Frecuencia"):
    """Grafica el mapa de calor de correlación cruzada."""
    plt.figure(figsize=(14, 8))
    
    # Usar escala de colores divergente centrada en 0
    im = plt.imshow(correlation_matrix, aspect='auto', origin='lower', 
                    cmap='RdBu_r', vmin=-1, vmax=1)
    
    plt.colorbar(im, label='Correlación Cruzada')
    
    # Configurar etiquetas de eje X (lags)
    n_lags = len(lags_ms)
    step = max(1, n_lags // 10)
    plt.xticks(np.arange(0, n_lags, step), 
               [f"{lags_ms[i]:.0f}" for i in range(0, n_lags, step)])
    plt.xlabel('Lag (ms)')
    
    # Configurar etiquetas de eje Y (frecuencias)
    plt.yticks(np.arange(len(freq_labels)), freq_labels)
    plt.ylabel('Banda de Frecuencia')
    
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_lines(correlation_matrix, lags_ms, freq_labels, title="Correlación Cruzada por Bandas de Frecuencia - Lineal"):
    """Grafica una línea por banda y la identifica con su rango de frecuencia."""
    plt.figure(figsize=(16, 8))
    
    # Colores para las diferentes líneas
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_labels)))
    
    # Graficar cada banda de frecuencia e identificarla al final de la línea
    for i, (freq_label, color) in enumerate(zip(freq_labels, colors)):
        plt.plot(lags_ms, correlation_matrix[i, :], 
                color=color, linewidth=1.5, alpha=0.8)
        
        # Colocar números con espaciado vertical artificial a la derecha
        final_x = lags_ms[-1]
        final_y = correlation_matrix[i, -1]
        
        # Espaciar verticalmente los números de manera uniforme
        n_channels = len(freq_labels)
        y_range = correlation_matrix.max() - correlation_matrix.min()
        if y_range == 0:
            y_range = 0.1  # Valor por defecto si todas las correlaciones son iguales
        
        # Posición vertical espaciada artificialmente
        artificial_y = correlation_matrix.min() + (i / (n_channels - 1)) * y_range * 1.5
        
        # Posición del texto fuera del área del gráfico
        text_x = final_x + 30  # 30ms después del final del gráfico
        
        # Dibujar línea conectora desde el punto final hasta la posición artificial
        plt.plot([final_x, text_x - 5], [final_y, artificial_y], 
                color=color, linewidth=1, alpha=0.5, linestyle='--')
        
        # Colocar la banda de frecuencia
        plt.text(text_x, artificial_y, freq_label, 
                color=color, fontsize=9, fontweight='bold',
                ha='left', va='center')
    
    # Línea horizontal en y=0
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Línea vertical en x=0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.xlabel('Lag (ms)', fontsize=12)
    plt.ylabel('Correlación Cruzada', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Ajustar límites del eje X para dar espacio a los números
    plt.xlim(lags_ms[0], lags_ms[-1] + 60)
    
    plt.tight_layout()
    
    return plt.gcf()

def main():
    print("=" * 70)
    print("CORRELACIÓN CRUZADA: CANAL 1 vs CANAL 17 POR BANDAS DE FRECUENCIA")
    print("=" * 70)
    
    # Configuración
    fs = 1000.0  # Frecuencia de muestreo
    max_lag_ms = 250  # Máximo desfase en ms
    
    # Definir bandas de frecuencia de 2Hz: 0-2, 1-3, 2-4, ..., 28-30 Hz
    freq_bands = [(i, i+2) for i in range(0, 29, 1)]
    
    print(f"Frecuencia de muestreo: {fs} Hz")
    print(f"Bandas de frecuencia: {len(freq_bands)} (de 0-2Hz a 28-30Hz)")
    print(f"Rango de lags: -{max_lag_ms}ms a +{max_lag_ms}ms")
    print()
    
    # Cargar datos
    try:
        print("Cargando archivos .npy...")
        signal_ch1 = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
        signal_ch17 = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
        print(f"Canal 1 cargado: {len(signal_ch1)} muestras ({len(signal_ch1)/fs:.1f} segundos)")
        print(f"Canal 17 cargado: {len(signal_ch17)} muestras ({len(signal_ch17)/fs:.1f} segundos)")
    except Exception as e:
        print(f"Error cargando archivos .npy: {e}")
        return
    
    # Igualar longitudes
    min_len = min(len(signal_ch1), len(signal_ch17))
    signal_ch1 = signal_ch1[:min_len]
    signal_ch17 = signal_ch17[:min_len]
    print(f"Señales igualadas a: {min_len} muestras ({min_len/fs:.1f} segundos)")
    print()
    
    # Calcular correlación cruzada por bandas
    print("Calculando correlación cruzada por bandas de frecuencia...")
    correlation_matrix, lags_ms, freq_labels = compute_cross_correlation_by_band(
        signal_ch1, signal_ch17, fs, freq_bands, max_lag_ms
    )
    
    print()
    print("Resultados:")
    print(f"  - Matriz de correlación cruzada: {correlation_matrix.shape}")
    print(f"  - Rango de correlación cruzada: {correlation_matrix.min():.3f} a {correlation_matrix.max():.3f}")
    print(f"  - Correlación cruzada promedio: {correlation_matrix.mean():.3f}")
    print()
    
    # Encontrar máximos por banda de frecuencia
    print("Análisis de máximos por banda de frecuencia:")
    for i, freq_label in enumerate(freq_labels):
        max_corr = np.max(correlation_matrix[i, :])
        max_lag = lags_ms[np.argmax(correlation_matrix[i, :])]
        min_corr = np.min(correlation_matrix[i, :])
        min_lag = lags_ms[np.argmin(correlation_matrix[i, :])]
        print(f"  {freq_label}: máximo = {max_corr:.3f} en lag = {max_lag:.1f} ms, mínimo = {min_corr:.3f} en lag = {min_lag:.1f} ms")
    print()
    
    # Guardar datos
    print("Guardando resultados...")
    
    # Guardar matriz como CSV
    df_correlation = pd.DataFrame(correlation_matrix, index=freq_labels, columns=lags_ms)
    df_correlation.to_csv("cross_correlation_ch1_ch17_bands_matrix.csv")
    print("  - cross_correlation_ch1_ch17_bands_matrix.csv")
    
    # Guardar matriz como NPY
    np.save("cross_correlation_ch1_ch17_bands_matrix.npy", correlation_matrix)
    print("  - cross_correlation_ch1_ch17_bands_matrix.npy")
    
    # Graficar mapa de calor
    print("Generando mapa de calor...")
    fig = plot_heatmap(correlation_matrix, lags_ms, freq_labels, 
                      title="Correlación Cruzada Canal 1 vs Canal 17 por Bandas de Frecuencia")
    plt.savefig("cross_correlation_ch1_ch17_bands_heatmap.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch1_ch17_bands_heatmap.png")
    plt.close()
    
    # Graficar líneas
    print("Generando gráfico lineal...")
    fig = plot_lines(correlation_matrix, lags_ms, freq_labels, 
                    title="Correlación Cruzada Canal 1 vs Canal 17 por Bandas de Frecuencia - Lineal")
    plt.savefig("cross_correlation_ch1_ch17_bands_lines.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch1_ch17_bands_lines.png")
    plt.close()
    
    print()
    print("=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()