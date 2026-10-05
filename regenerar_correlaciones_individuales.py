"""
Script para regenerar las correlaciones individuales con información de dirección del liderazgo.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from leer_mat3 import (
    load_mat_data,
    correlacion_cruzada_combinatoria_cpf_amy,
)

def main():
    print("=" * 80)
    print("REGENERANDO CORRELACIONES INDIVIDUALES CON DIRECCIÓN DEL LIDERAZGO")
    print("=" * 80)
    print()

    # Cargar datos
    print("Cargando datos del archivo .mat...")
    archivo_mat = "allChan_1kHz_clean.mat"
    matriz_eeg, Fs, config = load_mat_data(archivo_mat)
    print(f"Datos cargados: {matriz_eeg.shape} @ {Fs} Hz")
    print()

    # Crear mapeo de canales
    original_channel_indices = [idx for idx in range(32) if idx not in {28, 29}]
    original_to_col = {idx + 1: col for col, idx in enumerate(original_channel_indices)}
    print(f"Mapeo de canales: {len(original_to_col)} canales disponibles")
    print()

    # Generar correlaciones individuales CPF vs Amy
    print("=" * 80)
    print("Generando correlaciones individuales CPF (1-10) vs Amy (17-25)...")
    print("=" * 80)
    try:
        df_resultados = correlacion_cruzada_combinatoria_cpf_amy(
            matriz_eeg, Fs, original_to_col
        )
        print("[OK] Correlaciones individuales generadas exitosamente")
        print()
    except Exception as e:
        print(f"[ERROR] Error en correlaciones individuales: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 80)
    print("PROCESO COMPLETADO!")
    print("=" * 80)
    print()
    print("Las imágenes individuales ahora incluyen información de dirección del liderazgo.")
    print("Revisa el directorio: correlaciones_individuales/")

if __name__ == "__main__":
    main()
