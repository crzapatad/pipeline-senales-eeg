# EEG2 - Sistema de Análisis de Señales EEG

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación](#instalación)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Guía de Uso](#guía-de-uso)
6. [Funcionalidades Principales](#funcionalidades-principales)
7. [Ejecución de Scripts](#ejecución-de-scripts)
8. [Dashboard Interactivo](#dashboard-interactivo)
9. [Solución de Problemas](#solución-de-problemas)

---

## 🧠 Descripción del Proyecto

EEG2 es un sistema completo para el análisis de señales electroencefalográficas (EEG) que permite:

- **Análisis de sincronización de fase** mediante MRL (Mean Resultant Length)
- **Correlación cruzada de envolvente** entre diferentes regiones cerebrales
- **Análisis por bandas de frecuencia** (Delta, Theta, Alpha, Beta, Gamma)
- **Comparaciones estadísticas** con controles aleatorios
- **Visualización interactiva** mediante dashboard web
- **Procesamiento de datos** en formato .mat (HDF5) y .npy

El proyecto está diseñado para analizar la conectividad funcional entre regiones cerebrales como:
- **CPF** (Corteza Prefrontal)
- **Amy** (Amígdala)
- **Nacc** (Núcleo Accumbens)
- **Hyp** (Hipotálamo)

---

## 💻 Requisitos del Sistema

### Sistema Operativo
- Windows 10 o superior (recomendado)
- Linux (compatible)
- macOS (compatible)

### Software Requerido
- **Python 3.8 o superior** (recomendado 3.9-3.11)
- **Git** (opcional, para clonar el repositorio)
- **PowerShell** (en Windows)

### Hardware Recomendado
- Mínimo: 8 GB RAM
- Recomendado: 16 GB RAM o más (para análisis extensos)
- Espacio en disco: 5 GB (para datos y resultados)

---

## 🚀 Instalación

### Paso 1: Preparar el Entorno en Windows

Abre PowerShell como administrador y ejecuta:

```powershell
# Configurar política de ejecución para PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Paso 2: Clonar o Copiar el Proyecto

Si tienes acceso al repositorio Git:

```powershell
git clone <url-del-repositorio>
cd EEG2
```

Si tienes los archivos directamente:
- Copia la carpeta `EEG2` a tu ubicación deseada (ej: `C:\Users\TuUsuario\Desktop\EEG2`)
- Navega a la carpeta en PowerShell:

```powershell
cd C:\Users\TuUsuario\Desktop\EEG2
```

### Paso 3: Crear Entorno Virtual

**Si ya existe el entorno virtual (.venv):**

```powershell
# Activar el entorno virtual existente
& C:\Users\TuUsuario\Desktop\EEG2\.venv\Scripts\Activate.ps1
```

**Si necesitas crear un nuevo entorno virtual:**

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar el entorno virtual
& .\.venv\Scripts\Activate.ps1
```

*Nota: Si aparece un error de ejecución de scripts, ejecuta primero:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Paso 4: Instalar Dependencias

Con el entorno virtual activado (verás `(.venv)` al inicio de la línea de comandos):

```powershell
# Instalar dependencias desde requirements_dashboard.txt
pip install -r requirements_dashboard.txt

# Instalar dependencias adicionales (si se necesitan)
pip install h5py scikit-learn joblib openpyxl
```

Las dependencias principales incluyen:
- `streamlit>=1.61.0` - Dashboard interactivo
- `plotly>=6.0.0` - Gráficos interactivos
- `pandas>=2.0.0` - Manipulación de datos
- `numpy>=1.24.0` - Cálculos numéricos
- `scipy>=1.10.0` - Procesamiento de señales
- `matplotlib>=3.7.0` - Visualización
- `h5py` - Lectura de archivos .mat
- `scikit-learn` - PCA y preprocesamiento
- `joblib` - Procesamiento paralelo

### Paso 5: Verificar Instalación

```powershell
# Verificar que Python esté usando el entorno virtual
python --version

# Verificar paquetes instalados
pip list
```

---

## 📁 Estructura del Proyecto

```
EEG2/
├── .venv/                          # Entorno virtual (no versionar)
├── eeg_dashboard.py                # Dashboard interactivo (Streamlit)
├── leer_mat3.py                    # Librería principal de procesamiento
├── leer_mat3_.py                   # Librería extendida con funciones adicionales
├── analisis_correlacion_envolvente.py  # Análisis de correlación de envolvente
├── analisis_ventanas_especificas.py    # Análisis de ventanas temporales
├── ejecutar.py                     # Script de ejecución de análisis MRL
├── correlacion_envolvente_simple.py    # Correlación de envolvente simplificada
├── cross_correlation_heatmap_bands.py # Mapas de calor por bandas
├── mrl_offsets_6s.py              # Análisis MRL de 6 segundos
├── mrl_random_comparison_5s.py    # Comparación aleatoria de 5 segundos
├── run_significance_analysis.py   # Análisis de significancia estadística
├── quick_significance_test.py      # Prueba rápida de significancia
├── visualizador_psd_interactivo.py # Visualizador de PSD interactivo
├── visualizador_psd_mejorado.py   # Visualizador de PSD mejorado
├── requirements_dashboard.txt      # Dependencias del dashboard
├── RESUMEN_CORRECCIONES.md        # Historial de correcciones
├── allChan_1kHz_clean.mat         # Datos EEG de ejemplo (archivo grande)
├── canal_1_PFC_butterworth_0_30_n4.npy  # Datos pre-procesados PFC
├── canal_17_Amy_butterworth_0_30_n4.npy # Datos pre-procesados Amígdala
└── [Archivos de resultados generados]   # CSV, PNG, MAT, etc.
```

---

## 📖 Guía de Uso

### Preparación de Datos

El proyecto puede trabajar con dos tipos de datos:

#### 1. Archivos .mat (formato HDF5)
- Archivo principal: `allChan_1kHz_clean.mat`
- Contiene datos EEG de múltiples canales
- Estructura esperada:
  - `allChan_clean`: Matriz de datos EEG (canales × muestras)
  - `Fs`: Frecuencia de muestreo
  - `processing/channel_configuration`: Configuración de canales por región

#### 2. Archivos .npy (NumPy)
- Formato binario de NumPy
- Ejemplos: `canal_1_PFC_butterworth_0_30_n4.npy`
- Contienen señales pre-filtradas de canales individuales

#### 3. Datos Open Ephys (formato .continuous)
- Carpeta `LFP/` con archivos `.continuous`
- Lectura automática si no se encuentra archivo .mat

---

## 🔧 Funcionalidades Principales

### 1. Dashboard Interactivo (Streamlit)

El dashboard proporciona una interfaz gráfica para explorar y analizar datos EEG.

**Características:**
- Explorador de datos (.npy, .csv, .mat)
- Análisis MRL con lags variables
- Análisis por frecuencia
- Comparación de fases
- Configuración interactiva de parámetros

### 2. Análisis MRL (Mean Resultant Length)

Calcula la sincronización de fase entre dos señales EEG usando el Mean Resultant Length.

**Scripts principales:**
- `ejecutar.py` - Ejecución completa con controles aleatorios
- `mrl_offsets_6s.py` - Análisis de ventanas de 6 segundos
- `mrl_random_comparison_5s.py` - Comparación con controles aleatorios

### 3. Correlación Cruzada de Envolvente

Analiza la correlación entre las envolventes de las señales en diferentes bandas de frecuencia.

**Scripts:**
- `analisis_correlacion_envolvente.py` - Análisis completo con significancia
- `correlacion_envolvente_simple.py` - Versión simplificada
- `cross_correlation_heatmap_bands.py` - Mapas de calor por bandas

### 4. Análisis por Ventanas Temporales

Analiza segmentos específicos del registro EEG.

**Scripts:**
- `analisis_ventanas_especificas.py` - Análisis de ventanas personalizadas
- Configuración por defecto: 1-6 min y 11-16 min

### 5. Análisis de Significancia Estadística

Valida si las correlaciones observadas son significativas comparando con distribuciones aleatorias.

**Scripts:**
- `run_significance_analysis.py` - Análisis completo de significancia
- `quick_significance_test.py` - Prueba rápida

### 6. Visualización de PSD (Power Spectral Density)

Visualiza el espectro de potencia de las señales EEG.

**Scripts:**
- `visualizador_psd_interactivo.py` - Visualizador interactivo
- `visualizador_psd_mejorado.py` - Visualizador mejorado

---

## 🎯 Ejecución de Scripts

### Ejecutar el Dashboard Interactivo

```powershell
# Asegúrate de estar en el entorno virtual
& C:\Users\TuUsuario\Desktop\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar el dashboard
streamlit run eeg_dashboard.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Ejecutar Análisis de Correlación de Envolvente

```powershell
# Activar entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar análisis completo
python analisis_correlacion_envolvente.py
```

**Archivos generados:**
- `envelope_significance_analisis_completo_filtered_matrix.csv`
- `envelope_significance_analisis_completo_heatmap.png`
- `envelope_significance_analisis_completo_summary.csv`
- `envelope_significance_analisis_completo_freq_offsets.csv`

### Ejecutar Análisis MRL con Controles Aleatorios

```powershell
# Activar entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar análisis MRL
python ejecutar.py
```

**Configuración:**
- Editar las líneas 144-177 en `ejecutar.py` para ajustar parámetros
- Frecuencia de muestreo: 1000 Hz (configurable)
- Ventanas: 1-6 min y 11-16 min (configurable)
- Iteraciones aleatorias: 50 (configurable)

### Ejecutar Análisis de Ventanas Específicas

```powershell
# Activar entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar análisis de ventanas
python analisis_ventanas_especificas.py
```

### Ejecutar Análisis de Significancia

```powershell
# Activar entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar análisis de significancia completo
python run_significance_analysis.py

# Ejecutar prueba rápida
python quick_significance_test.py
```

### Ejecutar Visualizador PSD

```powershell
# Activar entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Ejecutar visualizador interactivo
python visualizador_psd_interactivo.py
```

---

## 🖥️ Dashboard Interactivo

### Iniciar el Dashboard

```powershell
streamlit run eeg_dashboard.py
```

### Páginas del Dashboard

#### 1. Data Explorer
- Cargar archivos .npy (señales EEG)
- Cargar archivos .csv (resultados de análisis)
- Visualizar previsualización de datos
- Ver estadísticas básicas

#### 2. MRL Analysis
- Seleccionar dos señales para comparar
- Configurar frecuencia central y ancho de banda
- Configurar rango de lags (desfases temporales)
- Seleccionar ventana temporal (inicio y duración)
- Visualizar MRL vs Lag
- Identificar lag óptimo de máxima sincronización

#### 3. Frequency Analysis
- Análisis sistemático por frecuencias (1-30 Hz)
- Identificar offset óptimo para cada frecuencia
- Visualizar curva frecuencia vs offset
- Progreso en tiempo real

#### 4. Phase Comparison
- Comparar fases instantáneas de dos señales
- Visualizar fases en el tiempo
- Configurar parámetros de filtrado

#### 5. Settings
- Configurar frecuencia de muestreo global
- Ajustar parámetros comunes

### Tips del Dashboard
- Los datos cargados se mantienen en la sesión
- Usa el sidebar para navegación rápida
- Los gráficos son interactivos (zoom, pan)
- Puedes descargar resultados desde la sesión

---

## 🔍 Parámetros de Configuración Comunes

### Frecuencias de Muestreo
- **1 kHz** (1000 Hz) - Frecuencia estándar del proyecto
- Configurable en cada script (variable `fs` o `Fs`)

### Bandas de Frecuencia EEG
- **Delta**: 0.5 - 4 Hz
- **Theta**: 4 - 8 Hz
- **Alpha**: 8 - 13 Hz
- **Beta**: 13 - 30 Hz
- **Gamma baja**: 30 - 50 Hz
- **Gamma alta**: 50 - 100 Hz

### Rangos de Lag (Desfases)
- **Estándar**: -250 ms a +250 ms
- **Conservador**: -50 ms a +50 ms (para pruebas)
- Configurable en parámetros `max_lag_ms`

### Ventanas Temporales
- **Ventana 1**: 1-6 minutos (período "Morada")
- **Ventana 2**: 11-16 minutos (período "Laberinto")
- Configurable en variables `start_min` y `duration_min`

### Iteraciones Aleatorias
- **Pruebas**: 3-10 iteraciones
- **Análisis completo**: 50-100 iteraciones
- Configurable en `n_iterations` o `N_ITERACIONES_AZAR`

---

## 🛠️ Solución de Problemas

### Error: "No se puede cargar el archivo porque la ejecución de scripts está deshabilitada"

**Solución:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Error: "ModuleNotFoundError: No module named 'XXX'"

**Solución:**
```powershell
# Asegúrate de estar en el entorno virtual
& C:\Users\TuUsuario\Desktop\EEG\EEG2\.venv\Scripts\Activate.ps1

# Reinstalar dependencias
pip install -r requirements_dashboard.txt
pip install h5py scikit-learn joblib openpyxl
```

### Error: "FileNotFoundError: No se encontró archivo .mat"

**Solución:**
- Verifica que el archivo `allChan_1kHz_clean.mat` esté en la carpeta
- O utiliza archivos .npy pre-filtrados:
  - `canal_1_PFC_butterworth_0_30_n4.npy`
  - `canal_17_Amy_butterworth_0_30_n4.npy`

### Error: El análisis tarda demasiado

**Solución:**
- Reduce el número de iteraciones aleatorias
- Reduce el rango de frecuencias
- Reduce el rango de lags
- Usa parámetros conservadores (ver `RESUMEN_CORRECCIONES.md`)

### Error: "ValueError: operands could not be broadcast together"

**Solución:**
- Verifica que las señales tengan la misma longitud
- Asegúrate de usar la misma ventana temporal para ambas señales
- Revisa el archivo `RESUMEN_CORRECCIONES.md` para soluciones específicas

### El dashboard no se abre en el navegador

**Solución:**
```powershell
# Ejecutar con puerto específico
streamlit run eeg_dashboard.py --server.port 8501

# O abrir manualmente en: http://localhost:8501
```

### Problemas de memoria (RAM insuficiente)

**Solución:**
- Reduce el tamaño de las ventanas temporales
- Procesa una frecuencia a la vez
- Cierra otras aplicaciones
- Considera usar un archivo de datos más pequeño

---

## 📚 Referencias Adicionales

### Documentación Interna
- `RESUMEN_CORRECCIONES.md` - Historial de correcciones y optimizaciones
- Comentarios en código fuente - Documentación detallada de funciones

### Archivos de Configuración
- `requirements_dashboard.txt` - Dependencias del dashboard
- Variables en scripts - Parámetros configurables al inicio de cada script

### Archivos de Datos de Ejemplo
- `allChan_1kHz_clean.mat` - Datos EEG completos
- `canal_1_PFC_butterworth_0_30_n4.npy` - Canal PFC pre-filtrado
- `canal_17_Amy_butterworth_0_30_n4.npy` - Canal Amígdala pre-filtrado

---

## 🤝 Contribuciones y Soporte

Para reportar problemas o solicitar nuevas funcionalidades:
1. Revisa el archivo `RESUMEN_CORRECCIONES.md` para soluciones conocidas
2. Verifica que todas las dependencias estén instaladas correctamente
3. Revisa los archivos de log generados (`.log`) para errores específicos

---

## 📝 Notas Importantes

1. **Entorno Virtual**: Siempre activa el entorno virtual antes de ejecutar cualquier script
2. **Datos**: Asegúrate de tener los archivos de datos (.mat o .npy) en la carpeta del proyecto
3. **Espacio**: Los análisis generan muchos archivos temporales; asegúrate de tener espacio en disco
4. **Tiempo**: Los análisis completos pueden tardar varias horas dependiendo del tamaño de los datos
5. **Backups**: Considera hacer copias de seguridad de tus datos antes de ejecutar análisis extensos

---

## 🎓 Flujo de Trabajo Recomendado

### Para Principiantes

1. **Iniciar con el Dashboard**
   ```powershell
   streamlit run eeg_dashboard.py
   ```
   - Explora los datos disponibles
   - Familiarízate con la interfaz

2. **Prueba Rápida**
   ```powershell
   python quick_significance_test.py
   ```
   - Ejecuta un análisis simple para verificar la instalación

3. **Análisis Completo**
   ```powershell
   python analisis_correlacion_envolvente.py
   ```
   - Ejecuta el análisis completo con tus datos

### Para Análisis Avanzados

1. **Ajustar Parámetros**
   - Edita los scripts para modificar frecuencias, ventanas, iteraciones
   - Revisa `RESUMEN_CORRECCIONES.md` para parámetros conservadores vs completos

2. **Procesamiento por Lotes**
   - Usa scripts específicos para diferentes tipos de análisis
   - Combina resultados usando el dashboard

3. **Validación Estadística**
   - Ejecuta análisis de significancia
   - Compara con controles aleatorios

---

## 📄 Licencia

Este proyecto es para uso académico y de investigación. Consulta con los autores para permisos de uso específicos.

---

**Última actualización:** 2026-10-05
**Versión:** 1.0
