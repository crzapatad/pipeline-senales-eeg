import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def generar_grafico_psd(frecuencias, potencia, titulo="PSD", guardar_imagen=True, nombre_archivo=None):
    """
    Genera un gráfico simple de PSD sin interactividad.
    
    Args:
        frecuencias: Array de frecuencias
        potencia: Array de densidad de potencia
        titulo: Título del gráfico
        guardar_imagen: Si True, guarda el gráfico como imagen
        nombre_archivo: Nombre personalizado para guardar (opcional)
    """
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Graficar el PSD
    ax.plot(frecuencias, potencia, 'b-', linewidth=1.5, label='PSD')
    
    # Configurar ejes
    ax.set_xlabel("Frecuencia (Hz)", fontsize=12)
    ax.set_ylabel("Densidad de potencia", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3000)
    
    # Configurar ticks personalizados
    ax.set_xticks(np.arange(0, 101, 2))
    ax.set_xticklabels(np.arange(0, 101, 2), rotation=45, fontsize=8)
    ax.set_yticks(np.arange(0, 3001, 100))
    ax.set_yticklabels(np.arange(0, 3001, 100), fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Extra space for rotated labels
    
    # Guardar imagen si se solicita
    if guardar_imagen:
        if nombre_archivo is None:
            nombre_archivo = f"psd_{titulo.replace(' ', '_')}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {nombre_archivo}")
    
    plt.show()
    plt.close()
    
    return fig

def cargar_datos_psd(archivo_npy=None, archivo_csv=None):
    """
    Carga datos de PSD desde archivo .npy o .csv
    
    Args:
        archivo_npy: ruta al archivo .npy con datos [frecuencias, potencia]
        archivo_csv: ruta al archivo .csv con columnas frecuencia, potencia
    
    Returns:
        frecuencias, potencia
    """
    if archivo_npy and os.path.exists(archivo_npy):
        datos = np.load(archivo_npy)
        if datos.shape[0] == 2:
            frecuencias = datos[0]
            potencia = datos[1]
        else:
            # Si es un solo array, generar frecuencias asumiendo 0-100 Hz
            potencia = datos
            frecuencias = np.linspace(0, 100, len(potencia))
        return frecuencias, potencia
    
    elif archivo_csv and os.path.exists(archivo_csv):
        df = pd.read_csv(archivo_csv)
        if len(df.columns) >= 2:
            frecuencias = df.iloc[:, 0].values
            potencia = df.iloc[:, 1].values
        else:
            raise ValueError("El CSV debe tener al menos 2 columnas")
        return frecuencias, potencia
    
    else:
        # Generar datos de ejemplo
        print("No se encontraron archivos, generando datos de ejemplo...")
        frecuencias = np.linspace(0, 100, 1000)
        # Generar PSD con varios picos
        potencia = (np.exp(-((frecuencias - 10)**2) / 50) + 
                   0.6 * np.exp(-((frecuencias - 25)**2) / 30) +
                   0.3 * np.exp(-((frecuencias - 45)**2) / 20) +
                   0.1 * np.random.randn(len(frecuencias)))
        return frecuencias, potencia

def main():
    print("=" * 60)
    print("VISUALIZADOR DE PSD")
    print("=" * 60)
    print()
    
    # Buscar archivos disponibles
    archivos_npy = [f for f in os.listdir('.') if f.endswith('.npy') and 'psd' in f.lower()]
    archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv') and 'psd' in f.lower()]
    
    print("Archivos disponibles:")
    if archivos_npy:
        print("  Archivos .npy:")
        for f in archivos_npy[:5]:
            print(f"    - {f}")
    if archivos_csv:
        print("  Archivos .csv:")
        for f in archivos_csv[:5]:
            print(f"    - {f}")
    
    print()
    print("Opciones:")
    print("1. Usar datos de ejemplo")
    print("2. Cargar archivo específico")
    
    opcion = input("Seleccione opción (1-2): ").strip()
    
    frecuencias = None
    potencia = None
    titulo = "PSD"
    
    if opcion == "2":
        archivo = input("Ingrese nombre del archivo: ").strip()
        if archivo.endswith('.npy'):
            frecuencias, potencia = cargar_datos_psd(archivo_npy=archivo)
            titulo = f"PSD - {os.path.basename(archivo)}"
        elif archivo.endswith('.csv'):
            frecuencias, potencia = cargar_datos_psd(archivo_csv=archivo)
            titulo = f"PSD - {os.path.basename(archivo)}"
        else:
            print("Formato no soportado, usando datos de ejemplo")
            frecuencias, potencia = cargar_datos_psd()
    else:
        frecuencias, potencia = cargar_datos_psd()
    
    print()
    print("Generando gráfico de PSD...")
    
    # Generar gráfico simple
    generar_grafico_psd(frecuencias, potencia, titulo)
    
    print("Proceso completado.")

if __name__ == "__main__":
    main()