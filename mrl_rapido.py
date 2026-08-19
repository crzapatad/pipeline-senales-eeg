import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
from itertools import combinations
from joblib import Parallel, delayed

def cargar_mat_eeg(filepath):
    """
    Carga archivos .mat compatibles con versiones antiguas y v7.3 (HDF5).
    """
    try:
        data_mat = sio.loadmat(filepath)
        keys = [k for k in data_mat.keys() if not k.startswith('__')]
        data = np.array(data_mat[keys[0]])
    except (ValueError, NotImplementedError):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "El archivo está en formato MATLAB v7.3. "
                "Por favor ejecuta en tu terminal: pip install h5py"
            )
        with h5py.File(filepath, 'r') as f:
            keys = [k for k in f.keys() if not k.startswith('#')]
            data = np.array(f[keys[0]])
            
    # Ajustar dimensiones a (n_canales, n_muestras)
    if data.shape[0] > data.shape[1]:
        data = data.T

    return data

def calcular_mrl_par_individual(par, fases_matriz, lags):
    ch1_idx, ch2_idx = par
    fase1 = fases_matriz[ch1_idx]
    fase2 = fases_matriz[ch2_idx]
    n_samples = len(fase1)
    
    mrl_lags = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag < 0:
            s1 = fase1[-lag:]
            s2 = fase2[:n_samples + lag]
        elif lag > 0:
            s1 = fase1[:n_samples - lag]
            s2 = fase2[lag:]
        else:
            s1 = fase1
            s2 = fase2
            
        diferencia_fase = np.exp(1j * (s1 - s2))
        mrl_lags[i] = np.abs(np.mean(diferencia_fase))
        
    return par, mrl_lags

def _se_solapan(inicio1, fin1, inicio2, fin2):
    return max(inicio1, inicio2) < min(fin1, fin2)

def save_lagged_mrl_5min_three_random_comparison(
    filepath, 
    output_filepath="resultado_mrl.mat",
    fs=500, 
    real_window_min=(11, 16), 
    lag_max_sec=0.1, 
    num_random_windows=1,  
    n_jobs=-1
):
    print(f"Cargando archivo: {filepath}...")
    eeg_data = cargar_mat_eeg(filepath)
    
    n_canales, total_muestras = eeg_data.shape
    duracion_total_sec = total_muestras / fs
    print(f"EEG cargado exitosamente: {n_canales} canales, {total_muestras} muestras ({duracion_total_sec/60:.2f} min).")
    
    # Detectar si la matriz ya contiene valores de fase (en radianes entre -pi y pi)
    if np.min(eeg_data) >= -np.pi - 0.5 and np.max(eeg_data) <= np.pi + 0.5:
        print("Los datos cargados ya contienen la fase instantánea. Omitiendo Hilbert...")
        fases_eeg = eeg_data
    else:
        print("Calculando transformada de Hilbert para extraer la fase...")
        fases_eeg = np.angle(hilbert(eeg_data, axis=1))
    
    win_sec = 300 
    real_start_sec = real_window_min[0] * 60
    real_end_sec = real_window_min[1] * 60
    
    if real_end_sec > duracion_total_sec:
        print(f"¡ADVERTENCIA! El archivo dura solo {duracion_total_sec/60:.2f} min.")
        print("Ajustando la ventana real para que quepa en el archivo...")
        real_end_sec = duracion_total_sec
        real_start_sec = max(0, real_end_sec - win_sec)
        print(f"Nueva ventana real: {real_start_sec/60:.2f} min a {real_end_sec/60:.2f} min.")
    
    random_windows = []
    intentos = 0
    
    np.random.seed(42)
    print(f"Buscando {num_random_windows} ventana(s) aleatoria(s)...")
    
    while len(random_windows) < num_random_windows:
        intentos += 1
        if intentos > 10000:
            print("El archivo es corto para buscar ventanas aleatorias extra sin cruces.")
            print("Se procesará únicamente la ventana real.")
            break
        
        candidato_inicio = np.random.uniform(0, max(0.1, duracion_total_sec - win_sec))
        candidato_fin = candidato_inicio + win_sec
        
        if _se_solapan(candidato_inicio, candidato_fin, real_start_sec, real_end_sec): continue
            
        conflicto = False
        for r_ini, r_fin in random_windows:
            if _se_solapan(candidato_inicio, candidato_fin, r_ini, r_fin):
                conflicto = True
                break
        if conflicto: continue
            
        random_windows.append((candidato_inicio, candidato_fin))

    ventanas_totales = {'Real': (real_start_sec, real_end_sec)}
    for idx, (r_ini, r_fin) in enumerate(random_windows, 1):
        ventanas_totales[f'Aleatoria_{idx}'] = (r_ini, r_fin)

    pares = list(combinations(range(n_canales), 2))
    lags = np.arange(-int(lag_max_sec * fs), int(lag_max_sec * fs) + 1)
    
    resultados_finales = {}
    for nombre_ventana, (t_start, t_end) in ventanas_totales.items():
        print(f"\n---> Calculando MRL ventana [{nombre_ventana}] en PARALELO...")
        fases_segmento = fases_eeg[:, int(t_start * fs):int(t_end * fs)]
        mrl_paralelo = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(calcular_mrl_par_individual)(par, fases_segmento, lags) for par in pares
        )
        
        matriz_mrl = np.zeros((len(pares), len(lags)))
        for idx, (par, mrl_vector) in enumerate(mrl_paralelo):
            matriz_mrl[idx, :] = mrl_vector
        resultados_finales[nombre_ventana] = matriz_mrl

    resultados_finales['lags_sec'] = lags / fs
    resultados_finales['pares'] = np.array(pares)
    sio.savemat(output_filepath, resultados_finales)
    print(f"\n¡Completado exitosamente! Archivo guardado en: {output_filepath}")

if __name__ == '__main__':
    save_lagged_mrl_5min_three_random_comparison(
        filepath="allChan_phase_instantanea.mat",
        output_filepath="resultado_mrl.mat",
        fs=500,
        real_window_min=(11, 16), 
        num_random_windows=3      # <-- Cambia este 1 por un 3
    )