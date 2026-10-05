import numpy as np
import sys
sys.path.append('.')

# Importar las nuevas funciones desde leer_mat3
from EEG2.leer_mat3_ import (
    circular_shift,
    compute_random_correlation_with_circular_shift,
    test_significance_envelope_correlation,
    compute_envelope_correlation_with_significance,
    plot_envelope_correlation_with_significance,
    analyze_envelope_correlation_custom
)

def test_basic_functions():
    """Prueba básica de las nuevas funciones"""
    print("=" * 60)
    print("PRUEBA DE FUNCIONES DE CORRELACIÓN DE ENVOLVENTE")
    print("=" * 60)
    
    # Generar señales de prueba
    fs = 1000.0
    duration = 20  # segundos (necesario para desplazamiento circular de 5s)
    t = np.linspace(0, duration, int(fs * duration))
    
    # Señales con correlación conocida en banda theta (4-8 Hz)
    signal_a = np.sin(2 * np.pi * 6 * t) + 0.3 * np.random.randn(len(t))
    signal_b = np.sin(2 * np.pi * 6 * (t - 0.01)) + 0.3 * np.random.randn(len(t))  # 10ms delay
    
    print(f"Señales generadas: {len(signal_a)} muestras ({duration}s)")
    print()
    
    # Prueba 1: circular_shift
    print("Prueba 1: circular_shift")
    test_signal = np.array([1, 2, 3, 4, 5])
    shifted = circular_shift(test_signal, 2)
    print(f"Original: {test_signal}")
    print(f"Desplazado 2: {shifted}")
    assert np.array_equal(shifted, np.array([4, 5, 1, 2, 3])), "Error en circular_shift"
    print("[OK] circular_shift funciona correctamente")
    print()
    
    # Prueba 2: compute_random_correlation_with_circular_shift
    print("Prueba 2: compute_random_correlation_with_circular_shift")
    envelope_a = np.abs(np.sin(2 * np.pi * 6 * t))
    envelope_b = np.abs(np.sin(2 * np.pi * 6 * (t - 0.01)))
    
    random_corrs = compute_random_correlation_with_circular_shift(
        envelope_a, envelope_b, n_random=10, min_shift_sec=5, fs=fs
    )
    print(f"Correlaciones aleatorias generadas: {len(random_corrs)}")
    print(f"Rango: {np.nanmin(random_corrs):.3f} a {np.nanmax(random_corrs):.3f}")
    assert len(random_corrs) == 10, "Error en número de correlaciones aleatorias"
    print("[OK] compute_random_correlation_with_circular_shift funciona correctamente")
    print()
    
    # Prueba 3: test_significance_envelope_correlation
    print("Prueba 3: test_significance_envelope_correlation (bilateral)")
    real_corr = 0.8
    is_sig, threshold_high, threshold_low, p_val_high, p_val_low = test_significance_envelope_correlation(
        real_corr, random_corrs, percentile=95
    )
    print(f"Correlación real: {real_corr}")
    print(f"Umbral 95%: {threshold_high:.3f}")
    print(f"Umbral 5%: {threshold_low:.3f}")
    print(f"Significativa: {is_sig}")
    print(f"P-valor (high): {p_val_high:.3f}")
    print(f"P-valor (low): {p_val_low:.3f}")
    print("[OK] test_significance_envelope_correlation funciona correctamente")
    print()
    
    # Prueba 4: compute_envelope_correlation_with_significance (versión rápida)
    print("Prueba 4: compute_envelope_correlation_with_significance (versión rápida)")
    df_results, significant_offsets, correlation_matrix = compute_envelope_correlation_with_significance(
        signal_a, signal_b, fs,
        center_freqs=np.arange(4, 8, dtype=float),  # Solo 4-7 Hz para prueba rápida
        lags_ms=np.arange(-50, 51, 1, dtype=float),  # Solo -50 a +50 ms
        n_random=10,  # Solo 10 iteraciones para prueba
        min_shift_sec=5
    )
    
    print(f"Resultados generados: {len(df_results)} filas")
    print(f"Columnas: {list(df_results.columns)}")
    print(f"Matriz de correlaciones: {correlation_matrix.shape}")
    print(f"Offsets significativos por frecuencia:")
    for label, lags in significant_offsets.items():
        print(f"  {label}: {len(lags)} offsets significativos")
    
    assert len(df_results) > 0, "Error: no se generaron resultados"
    assert "Significativa" in df_results.columns, "Error: falta columna Significativa"
    assert correlation_matrix.shape[0] == 4, "Error: dimensiones incorrectas de matriz"
    print("[OK] compute_envelope_correlation_with_significance funciona correctamente")
    print()
    
    print("=" * 60)
    print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_basic_functions()
        print("\nLas nuevas funciones están listas para usar.")
        print("Para ejecutar el análisis completo con tus datos:")
        print("python leer_mat3.py")
    except Exception as e:
        print(f"Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()