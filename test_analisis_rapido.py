import numpy as np
import sys
sys.path.append('.')

from EEG2.leer_mat3_ import analyze_envelope_correlation_custom

def test_analisis_rapido():
    """Prueba rápida del análisis actualizado"""
    print("=" * 60)
    print("PRUEBA RÁPIDA DEL ANÁLISIS ACTUALIZADO")
    print("=" * 60)
    
    # Generar señales de prueba
    fs = 1000.0
    duration = 30  # segundos (necesario para desplazamiento circular de 5s)
    t = np.linspace(0, duration, int(fs * duration))
    
    # Señales con correlación en banda theta
    signal_a = np.sin(2 * np.pi * 6 * t) + 0.3 * np.random.randn(len(t))
    signal_b = np.sin(2 * np.pi * 6 * (t - 0.01)) + 0.3 * np.random.randn(len(t))
    
    print(f"Señales generadas: {len(signal_a)} muestras ({duration}s)")
    print()
    
    # Ejecutar análisis con parámetros reducidos para prueba rápida
    print("Ejecutando análisis con parámetros reducidos para prueba...")
    print("  - Frecuencias: 4-7 Hz (en lugar de 1-29 Hz)")
    print("  - Offsets: -50 a +50 ms (en lugar de -250 a +250 ms)")
    print("  - Iteraciones: 10 (en lugar de 1000)")
    print()
    
    # Modificar temporalmente la función para usar parámetros reducidos
    import EEG2.leer_mat3_ as leer_mat3_
    original_func = leer_mat3_.compute_envelope_correlation_with_significance
    
    def wrapped_func(*args, **kwargs):
        kwargs['center_freqs'] = np.arange(4, 8, dtype=float)  # 4-7 Hz
        kwargs['lags_ms'] = np.arange(-50, 51, 1, dtype=float)  # -50 a +50 ms
        kwargs['n_random'] = 10  # 10 iteraciones
        return original_func(*args, **kwargs)
    
    leer_mat3_.compute_envelope_correlation_with_significance = wrapped_func
    
    try:
        df_results, significant_offsets, correlation_matrix, freq_significant_offsets = analyze_envelope_correlation_custom(
            signal_a, signal_b, fs, window_name="prueba_rapida"
        )
        
        print()
        print("=" * 60)
        print("PRUEBA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        print()
        print("Archivos que deberían haberse generado:")
        print("  - envelope_significance_prueba_rapida_filtered_matrix.csv")
        print("  - envelope_significance_prueba_rapida_filtered_matrix.npy")
        print("  - envelope_significance_prueba_rapida_heatmap.png")
        print("  - envelope_significance_prueba_rapida_heatmap_limited.png")
        print("  - envelope_significance_prueba_rapida_summary.csv")
        print("  - envelope_significance_prueba_rapida_freq_offsets.csv")
        print()
        print("Verifica que estos archivos existan en el directorio.")
        
    except Exception as e:
        print(f"Error en prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analisis_rapido()