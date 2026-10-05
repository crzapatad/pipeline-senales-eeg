import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.image as mpimg

def generar_grafico_psd_mejorado(frecuencias=None, potencia=None, imagen_path=None, titulo="PSD", guardar_imagen=True, nombre_archivo=None):
    """
    Genera un gráfico simple de PSD sin interactividad.
    
    Args:
        frecuencias: Array de frecuencias (opcional)
        potencia: Array de densidad de potencia (opcional)
        imagen_path: Ruta a imagen PNG existente (opcional)
        titulo: Título del gráfico
        guardar_imagen: Si True, guarda el gráfico como imagen
        nombre_archivo: Nombre personalizado para guardar (opcional)
    """
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if imagen_path and os.path.exists(imagen_path):
        # Modo imagen: cargar y mostrar la imagen existente
        try:
            img = mpimg.imread(imagen_path)
            ax.imshow(img, extent=[0, 100, 0, 3000], aspect='auto', alpha=0.8)
            ax.set_ylim(0, 3000)  # Escala para modo imagen
            print(f"Imagen cargada: {imagen_path}")
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            # Si falla, intentar con datos
            imagen_path = None
    
    if not imagen_path or not os.path.exists(imagen_path):
        # Modo datos: graficar desde arrays
        if frecuencias is None or potencia is None:
            # Generar datos de ejemplo
            frecuencias = np.linspace(0, 100, 1000)
            potencia = (np.exp(-((frecuencias - 10)**2) / 50) + 
                       0.6 * np.exp(-((frecuencias - 25)**2) / 30) +
                       0.3 * np.exp(-((frecuencias - 45)**2) / 20) +
                       0.1 * np.random.randn(len(frecuencias)))
        
        ax.plot(frecuencias, potencia, 'b-', linewidth=1.5, label='PSD')
        ax.legend()
    
    # Configurar ejes comunes
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
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Extra space for rotated labels
    
    # Guardar imagen si se solicita
    if guardar_imagen:
        if nombre_archivo is None:
            nombre_archivo = f"psd_mejorado_{titulo.replace(' ', '_')}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado como: {nombre_archivo}")
    
    plt.show()
    plt.close()
    
    return fig

def main():
    print("=" * 70)
    print("VISUALIZADOR DE PSD - VERSIÓN MEJORADA")
    print("=" * 70)
    print()
    
    # Buscar archivos PNG de PSD disponibles
    archivos_png = [f for f in os.listdir('.') if f.endswith('.png') and 'psd' in f.lower() and 'canal' in f.lower()]
    
    print("Archivos PNG de PSD disponibles:")
    if archivos_png:
        for i, f in enumerate(archivos_png[:10], 1):
            print(f"  {i}. {f}")
    else:
        print("  No se encontraron archivos PNG de PSD")
    
    print()
    print("Opciones:")
    print("1. Cargar imagen PNG existente")
    print("2. Usar datos numéricos (npy/csv)")
    print("3. Generar datos de ejemplo")
    
    opcion = input("Seleccione opción (1-3): ").strip()
    
    imagen_path = None
    frecuencias = None
    potencia = None
    titulo = "PSD"
    
    if opcion == "1" and archivos_png:
        num = int(input(f"Seleccione archivo (1-{min(10, len(archivos_png))}): ")) - 1
        imagen_path = archivos_png[num]
        titulo = f"PSD - {os.path.basename(imagen_path)}"
    elif opcion == "2":
        archivo = input("Ingrese nombre del archivo (npy/csv): ").strip()
        if archivo.endswith('.npy') and os.path.exists(archivo):
            datos = np.load(archivo)
            if datos.shape[0] == 2:
                frecuencias = datos[0]
                potencia = datos[1]
            else:
                potencia = datos
                frecuencias = np.linspace(0, 100, len(potencia))
            titulo = f"PSD - {os.path.basename(archivo)}"
        elif archivo.endswith('.csv') and os.path.exists(archivo):
            df = pd.read_csv(archivo)
            frecuencias = df.iloc[:, 0].values
            potencia = df.iloc[:, 1].values
            titulo = f"PSD - {os.path.basename(archivo)}"
        else:
            print("Archivo no encontrado, usando datos de ejemplo")
    else:
        print("Usando datos de ejemplo")
    
    print()
    print("Generando gráfico de PSD...")
    
    # Generar gráfico simple
    generar_grafico_psd_mejorado(frecuencias, potencia, imagen_path, titulo)
    
    print("Proceso completado.")

if __name__ == "__main__":
    main()