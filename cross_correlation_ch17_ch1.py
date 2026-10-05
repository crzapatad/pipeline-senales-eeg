import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import pandas as pd

def main():
    print("=" * 70)
    print("CORRELACIÓN CRUZADA: CANAL 17 vs CANAL 1")
    print("=" * 70)
    
    # Configuración
    fs = 1000.0  # Frecuencia de muestreo
    max_lag_ms = 250  # Máximo desfase en ms
    
    print(f"Frecuencia de muestreo: {fs} Hz")
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
    
    # Normalizar señales (z-score)
    signal_ch1_norm = (signal_ch1 - np.mean(signal_ch1)) / (np.std(signal_ch1) + 1e-10)
    signal_ch17_norm = (signal_ch17 - np.mean(signal_ch17)) / (np.std(signal_ch17) + 1e-10)
    
    # Limpiar NaN/Inf
    signal_ch1_norm = np.nan_to_num(signal_ch1_norm, nan=0.0, posinf=0.0, neginf=0.0)
    signal_ch17_norm = np.nan_to_num(signal_ch17_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Usar una ventana más pequeña para velocidad
    window_size = min(len(signal_ch1_norm), 100000)
    signal_ch1_win = signal_ch1_norm[:window_size]
    signal_ch17_win = signal_ch17_norm[:window_size]
    
    print("Calculando correlación cruzada...")
    cross_corr = correlate(signal_ch1_win, signal_ch17_win, mode='same', method='auto')
    
    # Extraer rango de lags
    max_lag_samples = int(max_lag_ms / 1000.0 * fs)
    center_idx = len(cross_corr) // 2
    start_idx = center_idx - max_lag_samples
    end_idx = center_idx + max_lag_samples + 1
    
    cross_corr_limited = cross_corr[start_idx:end_idx]
    
    # Normalizar
    n = len(signal_ch1_win)
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = lags_samples / fs * 1000.0
    normalization = n - np.abs(lags_samples)
    normalization[normalization == 0] = 1
    
    cross_corr_normalized = cross_corr_limited / normalization
    cross_corr_normalized = np.nan_to_num(cross_corr_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    print()
    print("Resultados:")
    print(f"  - Longitud de correlación cruzada: {len(cross_corr_normalized)}")
    print(f"  - Rango: {cross_corr_normalized.min():.4f} a {cross_corr_normalized.max():.4f}")
    print(f"  - Promedio: {cross_corr_normalized.mean():.4f}")
    print(f"  - Desviación estándar: {cross_corr_normalized.std():.4f}")
    
    # Encontrar máximo
    max_corr = np.max(cross_corr_normalized)
    max_lag = lags_ms[np.argmax(cross_corr_normalized)]
    min_corr = np.min(cross_corr_normalized)
    min_lag = lags_ms[np.argmin(cross_corr_normalized)]
    
    print(f"  - Máximo: {max_corr:.4f} en lag = {max_lag:.1f} ms")
    print(f"  - Mínimo: {min_corr:.4f} en lag = {min_lag:.1f} ms")
    print()
    
    # Guardar datos
    print("Guardando resultados...")
    
    # CSV con lags y correlación
    df_results = pd.DataFrame({
        'lag_ms': lags_ms,
        'cross_correlation': cross_corr_normalized
    })
    df_results.to_csv("cross_correlation_ch17_ch1.csv", index=False)
    print("  - cross_correlation_ch17_ch1.csv")
    
    # NPY
    np.save("cross_correlation_ch17_ch1.npy", cross_corr_normalized)
    print("  - cross_correlation_ch17_ch1.npy")
    
    # Guardar lags por separado
    np.save("cross_correlation_lags_ms.npy", lags_ms)
    print("  - cross_correlation_lags_ms.npy")
    
    # Generar gráficos
    print("Generando gráficos...")
    
    # Gráfico 1: Correlación cruzada vs lag
    plt.figure(figsize=(14, 6))
    plt.plot(lags_ms, cross_corr_normalized, linewidth=2, color='blue')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=max_lag, color='red', linestyle=':', alpha=0.7, label=f'Máximo: {max_corr:.3f} @ {max_lag:.1f}ms')
    plt.xlabel('Lag (ms)', fontsize=12)
    plt.ylabel('Correlación Cruzada', fontsize=12)
    plt.title('Correlación Cruzada: Canal 17 vs Canal 1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cross_correlation_ch17_ch1_plot.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch17_ch1_plot.png")
    plt.close()
    
    # Gráfico 2: Zoom alrededor del máximo
    zoom_range = 50  # ms
    zoom_mask = (lags_ms >= max_lag - zoom_range) & (lags_ms <= max_lag + zoom_range)
    lags_zoom = lags_ms[zoom_mask]
    corr_zoom = cross_corr_normalized[zoom_mask]
    
    plt.figure(figsize=(12, 6))
    plt.plot(lags_zoom, corr_zoom, linewidth=2, color='darkblue')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=max_lag, color='red', linestyle=':', linewidth=2, label=f'Máximo: {max_corr:.3f} @ {max_lag:.1f}ms')
    plt.xlabel('Lag (ms)', fontsize=12)
    plt.ylabel('Correlación Cruzada', fontsize=12)
    plt.title(f'Correlación Cruzada: Canal 17 vs Canal 1 (Zoom ±{zoom_range}ms)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cross_correlation_ch17_ch1_zoom.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch17_ch1_zoom.png")
    plt.close()
    
    # Gráfico 3: Envelope de la correlación cruzada
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(cross_corr_normalized))
    
    plt.figure(figsize=(14, 6))
    plt.plot(lags_ms, cross_corr_normalized, linewidth=1, color='blue', alpha=0.7, label='Correlación Cruzada')
    plt.plot(lags_ms, envelope, linewidth=2, color='red', alpha=0.8, label='Envolvente')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Lag (ms)', fontsize=12)
    plt.ylabel('Correlación Cruzada', fontsize=12)
    plt.title('Correlación Cruzada con Envolvente: Canal 17 vs Canal 1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cross_correlation_ch17_ch1_envelope.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch17_ch1_envelope.png")
    plt.close()
    
    # Gráfico 4: Comparación de señales originales
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Canal 1
    plt.subplot(2, 1, 1)
    time = np.arange(len(signal_ch1_win)) / fs
    plt.plot(time, signal_ch1_win, color='blue', linewidth=0.5)
    plt.title('Canal 1 (PFC) - Primeros 100 segundos', fontsize=12)
    plt.ylabel('Amplitud (normalizada)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Canal 17
    plt.subplot(2, 1, 2)
    plt.plot(time, signal_ch17_win, color='red', linewidth=0.5)
    plt.title('Canal 17 (Amígdala) - Primeros 100 segundos', fontsize=12)
    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Amplitud (normalizada)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cross_correlation_ch17_ch1_signals.png", dpi=300, bbox_inches='tight')
    print("  - cross_correlation_ch17_ch1_signals.png")
    plt.close()
    
    print()
    print("=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print()
    print("Resumen:")
    print(f"  - Máxima correlación cruzada: {max_corr:.4f} a {max_lag:.1f} ms")
    print(f"  - Esto indica que el canal 17 está {max_lag:.1f} ms desfasado respecto al canal 1")
    print(f"  - Signo positivo: las señales están en fase (correlación positiva)")
    print(f"  - Magnitud: {abs(max_corr):.4f} (escala -1 a 1)")

if __name__ == "__main__":
    main()