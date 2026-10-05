import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'C:\\Users\\zcris\\Desktop\\EEG\\EEG2')
from EEG2.leer_mat3_ import (
    compute_envelope_correlation_with_surrogates,
    plot_envelope_correlation_with_significance
)

# Generar datos de prueba simples
print("=== GENERANDO DATOS DE PRUEBA ===")
fs = 1000  # 1 kHz
duration = 10  # 10 segundos
t = np.linspace(0, duration, int(fs * duration))

# Señales sintéticas con alguna correlación
signal1 = np.sin(2 * np.pi * 8 * t) + 0.5 * np.random.normal(0, 1, len(t))
signal2 = np.sin(2 * np.pi * 8 * (t - 0.01)) + 0.5 * np.random.normal(0, 1, len(t))

# Extraer envolventes simples (usando valor absoluto como aproximación)
envelope1 = np.abs(signal1)
envelope2 = np.abs(signal2)

print(f"Señales generadas: {len(signal1)} muestras")
print(f"Envolventes: {len(envelope1)} muestras")

# Probar función de correlación con subrogados
print("\n=== PROBANDO FUNCIÓN DE SIGNIFICANCIA ===")
results = compute_envelope_correlation_with_surrogates(
    envelope1, envelope2, 
    n_surrogates=50,  # Solo 50 para prueba rápida
    min_shift_samples=None, 
    fs=fs
)

print(f"Correlación real: {results['real_correlation']:.4f}")
print(f"Valor p: {results['p_value']:.4f}")
print(f"Z-score: {results['z_score']:.4f}")
print(f"Significativo: {results['is_significant']}")

# Crear DataFrame de prueba para visualización
print("\n=== CREANDO DATAFRAME DE PRUEBA ===")
test_data = []
for freq in [1, 2, 3, 4, 5]:
    for lag in [-50, -25, 0, 25, 50]:
        test_data.append({
            "Ventana": f"{freq}Hz({freq-1}-{freq+1}Hz)",
            "Frecuencia_Hz": float(freq),
            "Lag_ms": float(lag),
            "Correlacion_Envolvente": np.random.uniform(-0.5, 0.8),
            "P_Value": np.random.uniform(0.01, 0.9),
            "Z_Score": np.random.uniform(-3, 3),
            "Null_Mean": np.random.uniform(-0.2, 0.2),
            "Null_Std": np.random.uniform(0.1, 0.3),
            "Is_Significant": np.random.choice([True, False])
        })

df_test = pd.DataFrame(test_data)
print(f"DataFrame creado: {df_test.shape}")

# Probar función de visualización
print("\n=== PROBANDO FUNCIÓN DE VISUALIZACIÓN ===")
try:
    plot_envelope_correlation_with_significance(df_test, "test")
    print("Visualización completada exitosamente")
except Exception as e:
    print(f"Error en visualización: {e}")
    import traceback
    traceback.print_exc()

print("\n=== PRUEBA COMPLETADA ===")
print("Verifica que se hayan generado los siguientes archivos:")
print("- envelope_correlation_with_significance_test_matrix.csv")
print("- envelope_correlation_with_significance_test_heatmap.png")
print("- envelope_correlation_with_significance_test_scatter.png")
print("- envelope_correlation_with_significance_test_zscores.png")
print("- envelope_correlation_with_significance_test_summary.csv")
print("- envelope_correlation_with_significance_test_complete.csv")