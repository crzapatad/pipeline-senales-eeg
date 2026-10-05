import numpy as np
import sys
sys.path.append('.')

from EEG2.leer_mat3_ import analyze_envelope_correlation_custom, load_mat_data

def ejecutar_analisis_completo():
    """Ejecuta el análisis completo según las especificaciones del usuario:
    
    1. Filtrar en bandas de 2Hz (0-2, 1-3, ..., 28-30 Hz)
    2. Calcular envolvente positiva
    3. Correlación entre envolventes en 500 offsets (-250 a 250 ms)
    4. Identificar correlaciones significativas
    5. Offset al azar mayor que 5 segundos (1000 veces, desplazamiento circular)
    6. Si correlación real > 95% de valores aleatorios, se guarda
    7. Para cada frecuencia, obtener offset donde correlación es significativa
    8. Gráfico frecuencia x correlación (significativas marcadas)
    9. Mapa de calor de offset con escala simétrica (-0.9 a +0.9)
    """
    
    print("=" * 70)
    print("ANÁLISIS DE CORRELACIÓN DE ENVOLVENTE CON SIGNIFICANCIA")
    print("Según especificaciones del usuario")
    print("=" * 70)
    print()
    
    # Configuración de archivos
    use_mat_file = False  # Cambiar a True para usar archivo .mat
    
    if use_mat_file:
        # Opción 2: Usar archivo .mat principal
        mat_file = "allChan_1kHz_clean.mat"
        canal_pfc_idx = 0  # Índice del canal PFC (ajustar según configuración)
        canal_amig_idx = 16  # Índice del canal Amígdala (ajustar según configuración)
        
        print("Configuración:")
        print(f"  - Archivo MAT: {mat_file}")
        print(f"  - Canal PFC: índice {canal_pfc_idx}")
        print(f"  - Canal Amígdala: índice {canal_amig_idx}")
        print()
        
        try:
            print("Cargando datos desde archivo .mat...")
            matriz_eeg, fs, config = load_mat_data(mat_file)
            print(f"EEG cargado: {matriz_eeg.shape} ({fs} Hz)")
            
            # Seleccionar canales específicos
            signal_cpf = matriz_eeg[canal_pfc_idx, :]
            signal_amig = matriz_eeg[canal_amig_idx, :]
            print(f"Canal PFC seleccionado: {len(signal_cpf)} muestras ({len(signal_cpf)/fs:.1f} segundos)")
            print(f"Canal Amígdala seleccionado: {len(signal_amig)} muestras ({len(signal_amig)/fs:.1f} segundos)")
            
        except Exception as e:
            print(f"Error cargando archivo .mat: {e}")
            return
    else:
        # Opción 1: Usar archivos .npy pre-filtrados
        cpf_file = "canal_1_PFC_butterworth_0_30_n4.npy"
        amig_file = "canal_17_Amy_butterworth_0_30_n4.npy"
        fs = 1000.0
        
        print("Configuración:")
        print(f"  - Archivo CPF: {cpf_file}")
        print(f"  - Archivo Amígdala: {amig_file}")
        print(f"  - Frecuencia de muestreo: {fs} Hz")
        print()
        
        try:
            print("Cargando datos desde archivos .npy...")
            signal_cpf = np.load(cpf_file)
            signal_amig = np.load(amig_file)
            print(f"CPF cargado: {len(signal_cpf)} muestras ({len(signal_cpf)/fs:.1f} segundos)")
            print(f"Amígdala cargada: {len(signal_amig)} muestras ({len(signal_amig)/fs:.1f} segundos)")
            
        except Exception as e:
            print(f"Error cargando archivos .npy: {e}")
            print("Intentando usar archivo .mat...")
            use_mat_file = True
            mat_file = "allChan_1kHz_clean.mat"
            canal_pfc_idx = 0
            canal_amig_idx = 16
            
            try:
                matriz_eeg, fs, config = load_mat_data(mat_file)
                signal_cpf = matriz_eeg[canal_pfc_idx, :]
                signal_amig = matriz_eeg[canal_amig_idx, :]
                print(f"EEG cargado desde .mat: {matriz_eeg.shape} ({fs} Hz)")
                print(f"Canal PFC: {len(signal_cpf)} muestras ({len(signal_cpf)/fs:.1f} segundos)")
                print(f"Canal Amígdala: {len(signal_amig)} muestras ({len(signal_amig)/fs:.1f} segundos)")
            except Exception as e2:
                print(f"Error también con archivo .mat: {e2}")
                return
    
    # Asegurar mismas longitudes (común para ambas opciones)
    min_len = min(len(signal_cpf), len(signal_amig))
    signal_cpf = signal_cpf[:min_len]
    signal_amig = signal_amig[:min_len]
    print(f"Señales igualadas a: {min_len} muestras ({min_len/fs:.1f} segundos)")
    print()
    
    # Verificar que hay suficiente duración para desplazamiento circular de 5s
    min_duration = 2 * 5  # mínimo 10 segundos para desplazamiento circular de 5s
    actual_duration = min_len / fs
    if actual_duration < min_duration:
        print(f"ADVERTENCIA: Duración insuficiente para desplazamiento circular de 5s")
        print(f"  Duración requerida: {min_duration}s, Duración actual: {actual_duration:.1f}s")
        print("  Se usará un desplazamiento mínimo reducido automáticamente.")
        # Ajustar min_shift_sec proporcionalmente
        new_min_shift = max(1, int(actual_duration / 4))  # usar 1/4 de la duración como mínimo, mínimo 1s
        print(f"  Nuevo desplazamiento mínimo: {new_min_shift} segundos")
        # Modificar temporalmente el parámetro en la función
        import EEG2.leer_mat3_ as leer_mat3_
        original_func = leer_mat3_.compute_envelope_correlation_with_significance
        def wrapped_func(*args, **kwargs):
            kwargs['min_shift_sec'] = new_min_shift
            return original_func(*args, **kwargs)
        leer_mat3_.compute_envelope_correlation_with_significance = wrapped_func
    
    print("Iniciando análisis completo...")
    print()
    
    try:
        # Ejecutar análisis con especificaciones exactas
        df_results, significant_offsets, correlation_matrix, freq_significant_offsets = analyze_envelope_correlation_custom(
            signal_cpf, signal_amig, fs, window_name="analisis_completo"
        )
        
        print()
        print("=" * 70)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print()
        print("Archivos generados:")
        print("  - envelope_significance_analisis_completo_filtered_matrix.csv")
        print("  - envelope_significance_analisis_completo_filtered_matrix.npy")
        print("  - envelope_significance_analisis_completo_heatmap.png (escala completa)")
        print("  - envelope_significance_analisis_completo_heatmap_limited.png (escala -0.9 a +0.9)")
        print("  - envelope_significance_analisis_completo_summary.csv")
        print("  - envelope_significance_analisis_completo_freq_offsets.csv")
        print()
        print("Resumen de resultados:")
        print(f"  - Total de correlaciones calculadas: {len(df_results)}")
        print(f"  - Correlaciones significativas: {len(df_results[df_results['Significativa']])}")
        print(f"  - Frecuencias con offsets significativos: {len([k for k, v in significant_offsets.items() if len(v) > 0])}")
        print(f"  - Matriz de correlaciones filtrada: {correlation_matrix.shape}")
        print()
        
        # Mostrar resumen de offsets significativos por frecuencia
        print("Offsets significativos por frecuencia:")
        for freq in sorted(freq_significant_offsets.keys()):
            offsets = freq_significant_offsets[freq]
            if len(offsets) > 0:
                print(f"  {freq} Hz: {len(offsets)} offsets {offsets[:3]}{'...' if len(offsets) > 3 else ''}")
            else:
                print(f"  {freq} Hz: Sin offsets significativos")
    
    except Exception as e:
        print(f"ERROR durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ejecutar_analisis_completo()