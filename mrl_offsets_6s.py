import os
from pathlib import Path

import h5py
import numpy as np

from leer_mat2 import save_lagged_mrl_6s_outputs


def load_fs(base_dir):
    for mat_path in (base_dir / "allChan_1kHz_clean.mat", base_dir.parent / "allChan_1kHz_clean.mat"):
        if mat_path.exists():
            with h5py.File(mat_path, "r") as data:
                return float(np.array(data["Fs"])[0][0])
    raise FileNotFoundError("No se encontro allChan_1kHz_clean.mat para leer Fs")


def main():
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    fs = load_fs(base_dir)
    residual_path = base_dir / "allChan_residual_reduced.npy"
    if not residual_path.exists():
        raise FileNotFoundError("No se encontro EEG2/allChan_residual_reduced.npy")

    residual = np.load(residual_path, mmap_mode="r")
    signal_ch1_pfc = np.asarray(residual[:, 0])
    signal_ch17_amy = np.asarray(residual[:, 16])

    window_specs = [
        ("1_6min", 60.0, 360.0),
        ("11_16min", 660.0, 960.0),
    ]

    for window_name, start_s, end_s in window_specs:
        save_lagged_mrl_6s_outputs(
            signal_ch1_pfc,
            signal_ch17_amy,
            fs,
            window_name,
            start_s,
            end_s,
            center_freqs=np.arange(1, 30, dtype=float),
            window_width_hz=2.0,
            lags_ms=np.arange(-250, 251, 1, dtype=float),
            short_window_s=6.0,
            short_window_step_s=3.0,
        )


if __name__ == "__main__":
    main()
