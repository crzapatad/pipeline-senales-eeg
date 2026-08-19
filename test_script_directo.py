import sys
print("Iniciando...", flush=True)
sys.stdout.flush()

import numpy as np
print("NumPy importado", flush=True)
sys.stdout.flush()

cpf = np.load("canal_1_PFC_butterworth_0_30_n4.npy")
print("CPF cargado", flush=True)
sys.stdout.flush()

amig = np.load("canal_17_Amy_butterworth_0_30_n4.npy")
print("Amígdala cargada", flush=True)
sys.stdout.flush()

print(f"Datos: CPF={cpf.shape}, Amígdala={amig.shape}", flush=True)
sys.stdout.flush()

print("Script completado", flush=True)
sys.stdout.flush()
