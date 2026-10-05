import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
import pandas as pd
import h5py
import os
from itertools import product

def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
    """Aplica filtro butterworth pasabanda"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, padlen=None)
    return filtered

def detect_data_source(target_fs=None):
    """Detecta y carga datos EEG desde archivo .mat"""
    if os.path.exists("allChan_1kHz_clean.mat"):
        print("Cargando datos desde allChan_1kHz_clean.mat")
        with h5py.File("allChan_1kHz_clean.mat", "r") as f:
            datos = np.array(f["allChan_clean"])
            Fs = 1000.0  # Frecuencia de muestreo asumida
            config = None
            return datos, Fs, config
    elif os.path.exists("allChan_residual_reduced.mat"):
        print("Cargando datos desde allChan_residual_reduced.mat")
        with h5py.File("allChan_residual_reduced.mat", "r") as f:
            datos = np.array(f["allChan_residual_reduced"])
            Fs = 1000.0  # Frecuencia de muestreo asumida
            config = None
            return datos, Fs, config
    elif os.path.exists("allChan_clean.mat"):
        print("Cargando datos desde allChan_clean.mat")
        with h5py.File("allChan_clean.mat", "r") as f:
            datos = np.array(f["allChan_clean"])
            Fs = 1000.0
            config = None
            return datos, Fs, config
    else:
        raise FileNotFoundError("No se encontraron archivos de datos EEG (.mat)")

def normalizar_senal(signal):
    """Normaliza señal usando z-score"""
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    signal_norm = np.nan_to_num(signal_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return signal_norm

def calcular_correlacion_cruzada(senal1, senal2, fs, max_lag_ms=250):
    """Calcula correlación cruzada entre dos señales"""
    # Normalizar señales
    senal1_norm = normalizar_senal(senal1)
    senal2_norm = normalizar_senal(senal2)
    
    # Usar ventana más pequeña para evitar overflow
    window_size = min(len(senal1_norm), 50000)  # Reducido a 50000 muestras (50s)
    senal1_win = senal1_norm[:window_size]
    senal2_win = senal2_norm[:window_size]
    
    # Calcular correlación cruzada
    cross_corr = correlate(senal1_win, senal2_win, mode='same', method='auto')
    
    # Extraer rango de lags
    max_lag_samples = int(max_lag_ms / 1000.0 * fs)
    center_idx = len(cross_corr) // 2
    start_idx = center_idx - max_lag_samples
    end_idx = center_idx + max_lag_samples + 1
    
    cross_corr_limited = cross_corr[start_idx:end_idx]
    
    # Normalizar
    n = len(senal1_win)
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    lags_ms = lags_samples / fs * 1000.0
    normalization = n - np.abs(lags_samples)
    normalization[normalization == 0] = 1
    
    cross_corr_normalized = cross_corr_limited / normalization
    cross_corr_normalized = np.nan_to_num(cross_corr_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encontrar máximo y mínimo
    max_corr = np.max(cross_corr_normalized)
    max_lag = lags_ms[np.argmax(cross_corr_normalized)]
    min_corr = np.min(cross_corr_normalized)
    min_lag = lags_ms[np.argmin(cross_corr_normalized)]
    
    return {
        'lags_ms': lags_ms,
        'cross_corr': cross_corr_normalized,
        'max_corr': max_corr,
        'max_lag': max_lag,
        'min_corr': min_corr,
        'min_lag': min_lag
    }

def main():
    print("=" * 80)
    print("CORRELACIÓN CRUZADA COMBINATORIA: CPF (1-10) vs AMÍGDALA (17-25)")
    print("=" * 80)
    print()
    
    # Configuración
    fs = 1000.0  # Frecuencia de muestreo
    max_lag_ms = 250  # Máximo desfase en ms
    filtro_low = 0.1  # Hz (cambiado de 0.0 para evitar error de scipy)
    filtro_high = 30.0  # Hz
    
    # Definir canales (números originales)
    canales_cpf = list(range(1, 11))  # 1-10
    canales_amy = list(range(17, 26))  # 17-25
    
    print(f"Canales CPF (cortex prefrontal): {canales_cpf}")
    print(f"Canales Amy (amígdala): {canales_amy}")
    print(f"Total combinaciones: {len(canales_cpf) * len(canales_amy)}")
    print()
    
    # Cargar datos
    try:
        print("Cargando datos EEG...")
        matriz_eeg, Fs, config = detect_data_source()
        print(f"Datos cargados: {matriz_eeg.shape}")
        print(f"Frecuencia de muestreo: {Fs} Hz")
        print()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
    
    # Crear mapeo de canales originales a columnas
    n_canales = matriz_eeg.shape[1]
    original_to_col = {i+1: i for i in range(n_canales)}
    
    # Verificar que los canales solicitados existan
    canales_faltantes = []
    for c in canales_cpf + canales_amy:
        if c not in original_to_col:
            canales_faltantes.append(c)
    
    if canales_faltantes:
        print(f"ERROR: Los siguientes canales no están disponibles: {canales_faltantes}")
        return
    
    print("Todos los canales solicitados están disponibles.")
    print()
    
    # Generar todas las combinaciones
    combinaciones = list(product(canales_cpf, canales_amy))
    print(f"Combinaciones a analizar: {len(combinaciones)}")
    print()
    
    # Almacenar resultados
    resultados = []
    datos_correlacion = {}
    
    print("Iniciando análisis de correlación cruzada...")
    print()
    
    for i, (canal_cpf, canal_amy) in enumerate(combinaciones, 1):
        print(f"[{i}/{len(combinaciones)}] Analizando CPF-{canal_cpf} vs Amy-{canal_amy}")
        
        # Obtener índices de columnas
        col_cpf = original_to_col[canal_cpf]
        col_amy = original_to_col[canal_amy]
        
        # Extraer señales
        senal_cpf = matriz_eeg[:, col_cpf]
        senal_amy = matriz_eeg[:, col_amy]
        
        # Igualar longitudes
        min_len = min(len(senal_cpf), len(senal_amy))
        senal_cpf = senal_cpf[:min_len]
        senal_amy = senal_amy[:min_len]
        
        # Calcular correlación cruzada directamente sin filtro
        resultado = calcular_correlacion_cruzada(senal_cpf, senal_amy, Fs, max_lag_ms)
        
        # Guardar resultado
        info_resultado = {
            'CPF_canal': canal_cpf,
            'Amy_canal': canal_amy,
            'max_corr': resultado['max_corr'],
            'max_lag_ms': resultado['max_lag'],
            'min_corr': resultado['min_corr'],
            'min_lag_ms': resultado['min_lag'],
            'mean_corr': np.mean(resultado['cross_corr']),
            'std_corr': np.std(resultado['cross_corr'])
        }
        resultados.append(info_resultado)
        
        # Guardar datos de correlación para visualización
        key = f"CPF_{canal_cpf}_Amy_{canal_amy}"
        datos_correlacion[key] = resultado
        
        print(f"  - Máx: {resultado['max_corr']:.4f} @ {resultado['max_lag']:.1f}ms")
        print(f"  - Mín: {resultado['min_corr']:.4f} @ {resultado['min_lag']:.1f}ms")
        print()
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Guardar resultados
    print("Guardando resultados...")
    df_resultados.to_csv("correlacion_cruzada_combinatoria_resumen.csv", index=False)
    print("  - correlacion_cruzada_combinatoria_resumen.csv")
    
    # Guardar datos de correlación completos
    np.save("correlacion_cruzada_combinatoria_datos.npy", datos_correlacion)
    print("  - correlacion_cruzada_combinatoria_datos.npy")
    
    # Generar directorio para resultados individuales
    directorio_individual = "correlaciones_individuales"
    if not os.path.exists(directorio_individual):
        os.makedirs(directorio_individual)
        print(f"Directorio creado: {directorio_individual}")
    
    # Generar gráficos y CSVs individuales para cada combinación
    print()
    print("Generando gráficos y CSVs individuales...")
    
    for i, (canal_cpf, canal_amy) in enumerate(combinaciones, 1):
        key = f"CPF_{canal_cpf}_Amy_{canal_amy}"
        if key in datos_correlacion:
            data = datos_correlacion[key]
            
            # Generar gráfico individual
            plt.figure(figsize=(12, 5))
            plt.plot(data['lags_ms'], data['cross_corr'], linewidth=2, color='blue')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=data['max_lag'], color='red', linestyle=':', linewidth=2, 
                       label=f'Máx: {data["max_corr"]:.3f} @ {data["max_lag"]:.1f}ms')
            plt.axvline(x=data['min_lag'], color='green', linestyle=':', linewidth=2, 
                       label=f'Mín: {data["min_corr"]:.3f} @ {data["min_lag"]:.1f}ms')
            
            plt.xlabel('Lag (ms)', fontsize=12)
            plt.ylabel('Correlación Cruzada', fontsize=12)
            plt.title(f'Correlación Cruzada: CPF-{canal_cpf} vs Amy-{canal_amy}', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            nombre_grafico = f"{directorio_individual}/correlacion_CPF{canal_cpf}_Amy{canal_amy}.png"
            plt.savefig(nombre_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generar CSV individual
            df_individual = pd.DataFrame({
                'lag_ms': data['lags_ms'],
                'cross_correlation': data['cross_corr']
            })
            nombre_csv = f"{directorio_individual}/correlacion_CPF{canal_cpf}_Amy{canal_amy}.csv"
            df_individual.to_csv(nombre_csv, index=False)
            
            if i % 10 == 0:
                print(f"  Progreso: {i}/{len(combinaciones)} combinaciones procesadas")
    
    print(f"  Completado: {len(combinaciones)} gráficos y CSVs individuales generados")
    
    # Generar visualizaciones resumen
    print()
    print("Generando visualizaciones resumen...")
    
    # 1. Heatmap de correlaciones máximas
    plt.figure(figsize=(12, 8))
    heatmap_data = df_resultados.pivot(index='CPF_canal', columns='Amy_canal', values='max_corr')
    
    # Usar imshow en lugar de seaborn
    im = plt.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlación Máxima')
    
    # Añadir anotaciones
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = plt.text(j, i, f'{heatmap_data.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.title('Correlación Cruzada Máxima: CPF vs Amígdala', fontsize=14, fontweight='bold')
    plt.xlabel('Canal Amígdala', fontsize=12)
    plt.ylabel('Canal CPF', fontsize=12)
    plt.tight_layout()
    plt.savefig("correlacion_cruzada_heatmap_max.png", dpi=300, bbox_inches='tight')
    print("  - correlacion_cruzada_heatmap_max.png")
    plt.close()
    
    # 2. Heatmap de lags de correlación máxima
    plt.figure(figsize=(12, 8))
    heatmap_lag = df_resultados.pivot(index='CPF_canal', columns='Amy_canal', values='max_lag_ms')
    
    # Usar imshow en lugar de seaborn
    im2 = plt.imshow(heatmap_lag.values, cmap='viridis', aspect='auto')
    plt.colorbar(im2, label='Lag (ms)')
    
    # Añadir anotaciones
    for i in range(len(heatmap_lag.index)):
        for j in range(len(heatmap_lag.columns)):
            text = plt.text(j, i, f'{heatmap_lag.values[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=8)
    
    plt.xticks(range(len(heatmap_lag.columns)), heatmap_lag.columns)
    plt.yticks(range(len(heatmap_lag.index)), heatmap_lag.index)
    plt.title('Lag de Correlación Máxima (ms): CPF vs Amígdala', fontsize=14, fontweight='bold')
    plt.xlabel('Canal Amígdala', fontsize=12)
    plt.ylabel('Canal CPF', fontsize=12)
    plt.tight_layout()
    plt.savefig("correlacion_cruzada_heatmap_lag.png", dpi=300, bbox_inches='tight')
    print("  - correlacion_cruzada_heatmap_lag.png")
    plt.close()
    
    # 3. Gráfico de barras de correlaciones máximas
    plt.figure(figsize=(16, 6))
    x_positions = np.arange(len(df_resultados))
    plt.bar(x_positions, df_resultados['max_corr'], color='steelblue', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Etiquetas de eje X
    labels = [f"CPF{row['CPF_canal']}-Amy{row['Amy_canal']}" for _, row in df_resultados.iterrows()]
    plt.xticks(x_positions, labels, rotation=90, fontsize=8)
    
    plt.ylabel('Correlación Máxima', fontsize=12)
    plt.title('Correlación Cruzada Máxima por Par de Canales', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("correlacion_cruzada_barras_max.png", dpi=300, bbox_inches='tight')
    print("  - correlacion_cruzada_barras_max.png")
    plt.close()
    
    # 4. Scatter plot: correlación vs lag
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_resultados['max_lag_ms'], df_resultados['max_corr'], 
                        c=range(len(df_resultados)), cmap='viridis', s=50, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Lag de Correlación Máxima (ms)', fontsize=12)
    plt.ylabel('Correlación Máxima', fontsize=12)
    plt.title('Relación entre Correlación y Lag', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Índice de combinación')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlacion_cruzada_scatter.png", dpi=300, bbox_inches='tight')
    print("  - correlacion_cruzada_scatter.png")
    plt.close()
    
    # 5. Top 10 combinaciones con mayor correlación
    top_10 = df_resultados.nlargest(10, 'max_corr')
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(top_10))
    plt.bar(x_pos, top_10['max_corr'], color='darkgreen', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    labels = [f"CPF{row['CPF_canal']}-Amy{row['Amy_canal']}" for _, row in top_10.iterrows()]
    plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=10)
    
    plt.ylabel('Correlación Máxima', fontsize=12)
    plt.title('Top 10 Combinaciones con Mayor Correlación', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("correlacion_cruzada_top10.png", dpi=300, bbox_inches='tight')
    print("  - correlacion_cruzada_top10.png")
    plt.close()
    
    print()
    print("=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print()
    print("Resumen estadístico:")
    print(df_resultados.describe())
    print()
    print("Top 10 combinaciones con mayor correlación:")
    print(top_10[['CPF_canal', 'Amy_canal', 'max_corr', 'max_lag_ms']].to_string(index=False))
    print()
    print("Archivos generados:")
    print("  - correlacion_cruzada_combinatoria_resumen.csv (resumen general)")
    print("  - correlacion_cruzada_combinatoria_datos.npy (datos completos)")
    print("  - correlacion_cruzada_heatmap_max.png")
    print("  - correlacion_cruzada_heatmap_lag.png")
    print("  - correlacion_cruzada_barras_max.png")
    print("  - correlacion_cruzada_scatter.png")
    print("  - correlacion_cruzada_top10.png")
    print(f"  - {len(combinaciones)} gráficos individuales en directorio 'correlaciones_individuales/'")
    print(f"  - {len(combinaciones)} archivos CSV individuales en directorio 'correlaciones_individuales/'")

if __name__ == "__main__":
    main()