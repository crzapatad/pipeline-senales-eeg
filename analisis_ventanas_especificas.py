import numpy as np
import sys
sys.path.append('.')

from EEG2.leer_mat3_ import analyze_envelope_correlation_custom, load_mat_data

def analizar_ventana_especifica(signal_cpf, signal_amig, fs, ventana_nombre, start_min, dur_min):
    """Analiza una ventana de tiempo específica"""
    print("=" * 70)
    print(f"ANALIZANDO VENTANA: {ventana_nombre}")
    print(f"Tiempo: {start_min} - {start_min + dur_min} minutos")
    print("=" * 70)
    
    # Extraer ventana específica
    start_idx = int(start_min * 60 * fs)
    dur_samples = int(dur_min * 60 * fs)
    
    cpf_window = signal_cpf[start_idx:start_idx + dur_samples]
    amig_window = signal_amig[start_idx:start_idx + dur_samples]
    
    print(f"Ventana extraída: CPF={len(cpf_window)} muestras, Amígdala={len(amig_window)} muestras")
    print(f"Duración: {dur_min} minutos ({dur_min*60} segundos)")
    print()
    
    # Verificar duración suficiente
    actual_duration = len(cpf_window) / fs
    if actual_duration < 10:  # mínimo 10 segundos para desplazamiento de 5s
        print(f"ADVERTENCIA: Ventana demasiado corta ({actual_duration:.1f}s < 10s mínimo)")
        print("Se usará desplazamiento mínimo reducido.")
        new_min_shift = max(1, int(actual_duration / 4))
        print(f"Nuevo desplazamiento mínimo: {new_min_shift} segundos")
        
        import EEG2.leer_mat3_ as leer_mat3_
        original_func = leer_mat3_.compute_envelope_correlation_with_significance
        def wrapped_func(*args, **kwargs):
            kwargs['min_shift_sec'] = new_min_shift
            return original_func(*args, **kwargs)
        leer_mat3_.compute_envelope_correlation_with_significance = wrapped_func
    
    # Ejecutar análisis
    window_id = f"{ventana_nombre}_{start_min}_{start_min+dur_min}min"
    df_results, significant_offsets, correlation_matrix, freq_significant_offsets = analyze_envelope_correlation_custom(
        cpf_window, amig_window, fs, window_name=window_id
    )
    
    return df_results, significant_offsets, correlation_matrix, freq_significant_offsets

def ejecutar_analisis_ventanas():
    """Ejecuta análisis para ventanas específicas (Morada y Laberinto)"""
    print("=" * 70)
    print("ANÁLISIS DE CORRELACIÓN DE ENVOLVENTE POR VENTANAS")
    print("Ventanas específicas: Morada (1-6 min) y Laberinto (11-16 min)")
    print("=" * 70)
    print()
    
    # Configuración de archivos
    use_mat_file = False  # Cambiar a True para usar archivo .mat
    
    if use_mat_file:
        # Usar archivo .mat
        mat_file = "allChan_1kHz_clean.mat"
        canal_pfc_idx = 0
        canal_amig_idx = 16
        
        print("Cargando datos desde archivo .mat...")
        matriz_eeg, fs, config = load_mat_data(mat_file)
        print(f"EEG cargado: {matriz_eeg.shape} ({fs} Hz)")
        
        signal_cpf = matriz_eeg[canal_pfc_idx, :]
        signal_amig = matriz_eeg[canal_amig_idx, :]
    else:
        # Usar archivos .npy
        cpf_file = "canal_1_PFC_butterworth_0_30_n4.npy"
        amig_file = "canal_17_Amy_butterworth_0_30_n4.npy"
        fs = 1000.0
        
        print("Cargando datos desde archivos .npy...")
        try:
            signal_cpf = np.load(cpf_file)
            signal_amig = np.load(amig_file)
            print(f"CPF: {len(signal_cpf)} muestras")
            print(f"Amígdala: {len(signal_amig)} muestras")
        except Exception as e:
            print(f"Error con .npy: {e}")
            print("Intentando con .mat...")
            matriz_eeg, fs, config = load_mat_data("allChan_1kHz_clean.mat")
            signal_cpf = matriz_eeg[0, :]
            signal_amig = matriz_eeg[16, :]
    
    # Igualar longitudes
    min_len = min(len(signal_cpf), len(signal_amig))
    signal_cpf = signal_cpf[:min_len]
    signal_amig = signal_amig[:min_len]
    print(f"Señales igualadas: {min_len} muestras ({min_len/fs:.1f} segundos)")
    print()
    
    # Definir ventanas a analizar
    ventanas = [
        ("Morada", 1, 6),      # Minuto 1 al 6
        ("Laberinto", 11, 16)  # Minuto 11 al 16
    ]
    
    resultados_completos = {}
    
    for nombre, start_min, dur_min in ventanas:
        try:
            df_results, significant_offsets, correlation_matrix, freq_offsets = analizar_ventana_especifica(
                signal_cpf, signal_amig, fs, nombre, start_min, dur_min
            )
            resultados_completos[nombre] = {
                'df': df_results,
                'significant_offsets': significant_offsets,
                'correlation_matrix': correlation_matrix,
                'freq_offsets': freq_offsets
            }
            print(f"Análisis de {nombre} completado exitosamente")
            print()
        except Exception as e:
            print(f"Error en análisis de {nombre}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Resumen final
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    for nombre, resultados in resultados_completos.items():
        df = resultados['df']
        sig_offsets = resultados['significant_offsets']
        correlation_matrix = resultados['correlation_matrix']
        freq_offsets = resultados['freq_offsets']
        
        print(f"\n{nombre}:")
        print(f"  - Correlaciones totales: {len(df)}")
        print(f"  - Correlaciones significativas: {len(df[df['Significativa']])}")
        print(f"  - Frecuencias con offsets significativos: {len([k for k, v in sig_offsets.items() if len(v) > 0])}")
        print(f"  - Matriz de correlaciones filtrada: {correlation_matrix.shape}")
        
        print(f"  - Offsets significativos por frecuencia:")
        for freq in sorted(freq_offsets.keys()):
            offsets = freq_offsets[freq]
            if len(offsets) > 0:
                print(f"    {freq} Hz: {len(offsets)} offsets {offsets[:2]}{'...' if len(offsets) > 2 else ''}")
            else:
                print(f"    {freq} Hz: Sin offsets significativos")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    ejecutar_analisis_ventanas()