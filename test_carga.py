import numpy as np
import time

print("Iniciando prueba de carga de archivos...")

# Prueba de carga de archivos
start = time.time()
try:
    cpf = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
    print(f"CPF cargado: {cpf.shape}, tipo: {cpf.dtype}")
except Exception as e:
    print(f"Error cargando CPF: {e}")

end = time.time()
print(f"Tiempo de carga CPF: {end - start:.2f} segundos")

start = time.time()
try:
    amig = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
    print(f"Amígdala cargada: {amig.shape}, tipo: {amig.dtype}")
except Exception as e:
    print(f"Error cargando Amígdala: {e}")

end = time.time()
print(f"Tiempo de carga Amígdala: {end - start:.2f} segundos")

print("Prueba completada.")
