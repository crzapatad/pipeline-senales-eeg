"""
Script para regenerar todas las imágenes con información de dirección del liderazgo.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from leer_mat3_ import (
    load_mat_data,
    correlacion_cruzada_combinatoria_cpf_amy,
    correlacion_cruzada_canal_1_17_primera_combinacion,
    correlacion_cruzada_canal_1_17_segunda_combinacion,
    correlacion_cruzada_canal_1_17_seis_combinaciones,
)

def main():
    print("=" * 80)
    print("REGENERANDO IMÁGENES CON INFORMACIÓN DE DIRECCIÓN DEL LIDERAZGO")
    print("=" * 80)
    print()

    # Cargar datos
    print("Cargando datos del archivo .mat...")
    archivo_mat = "allChan_1kHz_clean.mat"
    matriz_eeg, Fs, config = load_mat_data(archivo_mat)
    print(f"Datos cargados: {matriz_eeg.shape} @ {Fs} Hz")
    print()

    # Crear mapeo de canales
    # Basado en el código, necesitamos un mapeo original_to_col
    # Por defecto, usamos canales 1-32 (excluyendo 28, 29 según el código)
    original_channel_indices = [idx for idx in range(32) if idx not in {28, 29}]
    original_to_col = {idx + 1: col for col, idx in enumerate(original_channel_indices)}
    print(f"Mapeo de canales: {len(original_to_col)} canales disponibles")
    print()

    # 1. Generar gráficos combinatorios CPF vs Amy (grupo_bandas_*.png y todas_las_curvas_promedio_bandas_IC95.png)
    # YA COMPLETADO - Saltando este paso
    print("=" * 80)
    print("1. Gráficos combinatorios CPF vs Amy - YA GENERADOS (78 graficos)")
    print("=" * 80)
    print()

    # 2. Generar gráficos individuales canal 1 vs 17 - primera combinación
    print("=" * 80)
    print("2. Generando gráfico canal 1 vs 17 - primera combinación...")
    print("=" * 80)
    try:
        correlacion_cruzada_canal_1_17_primera_combinacion(
            matriz_eeg, Fs, original_to_col
        )
        print("[OK] Primera combinación generada exitosamente")
        print()
    except Exception as e:
        print(f"[ERROR] Error en primera combinación: {e}")
        print()

    # 3. Generar gráficos individuales canal 1 vs 17 - segunda combinación
    print("=" * 80)
    print("3. Generando gráfico canal 1 vs 17 - segunda combinación...")
    print("=" * 80)
    try:
        correlacion_cruzada_canal_1_17_segunda_combinacion(
            matriz_eeg, Fs, original_to_col
        )
        print("[OK] Segunda combinación generada exitosamente")
        print()
    except Exception as e:
        print(f"[ERROR] Error en segunda combinación: {e}")
        print()

    # 4. Generar gráficos individuales canal 1 vs 17 - seis combinaciones
    print("=" * 80)
    print("4. Generando gráficos canal 1 vs 17 - seis combinaciones...")
    print("=" * 80)
    try:
        correlacion_cruzada_canal_1_17_seis_combinaciones(
            matriz_eeg, Fs, original_to_col
        )
        print("[OK] Seis combinaciones generadas exitosamente")
        print()
    except Exception as e:
        print(f"[ERROR] Error en seis combinaciones: {e}")
        print()

    print("=" * 80)
    print("¡PROCESO COMPLETADO!")
    print("=" * 80)
    print()
    print("Las imágenes ahora incluyen información de dirección del liderazgo temporal.")
    print("Revisa los directorios:")
    print("  - todacombinacion_bandas/ (grupo_bandas_*.png)")
    print("  - correlaciones_individuales/ (si se ejecutó desde leer_mat3.py)")
    print("  - Directorio actual (correlacion_cruzada_canal_1_17_*.png)")

if __name__ == "__main__":
    main()
