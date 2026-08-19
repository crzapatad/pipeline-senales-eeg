print("Prueba de importaciones...")

try:
    import numpy as np
    print("NumPy importado")
except Exception as e:
    print(f"Error importando NumPy: {e}")

try:
    from scipy.signal import butter, filtfilt, hilbert
    print("scipy.signal importado")
except Exception as e:
    print(f"Error importando scipy.signal: {e}")

try:
    from typing import Tuple, Dict, List
    print("typing importado")
except Exception as e:
    print(f"Error importando typing: {e}")

try:
    import warnings
    warnings.filterwarnings('ignore')
    print("warnings importado")
except Exception as e:
    print(f"Error importando warnings: {e}")

print("Todas las importaciones completadas")
