import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'C:\\Users\\zcris\\Desktop\\EEG\\EEG2')
from EEG2.leer_mat3_ import (
    load_mat_data,
    run_envelope_correlation_significance_analysis
)

def main():
    print("=== EEG STATISTICAL SIGNIFICANCE ANALYSIS ===")
    print("This script analyzes connectivity between PFC and AMY channels")
    print("using envelope correlation with significance testing.")
    print()
    
    try:
        # Load data
        print("=== LOADING DATA ===")
        matriz_eeg, Fs, config = load_mat_data("allChan_1kHz_clean.mat")
        
        print(f"[OK] Data loaded: {matriz_eeg.shape}")
        print(f"[OK] Sampling frequency: {Fs} Hz")
        print(f"[OK] Total duration: {matriz_eeg.shape[0]/Fs/60:.1f} minutes")
        
        # Select channels (adjust according to your configuration)
        signal_pfc = matriz_eeg[:, 0]  # First channel (PFC)
        signal_amy = matriz_eeg[:, 16]  # Channel 17 (AMY)
        
        print(f"[OK] Channels selected: PFC (channel 1), AMY (channel 17)")
        
        # Analysis parameters (optimized for reasonable time)
        print("\n=== ANALYSIS CONFIGURATION ===")
        print("Frequencies: 1-30 Hz (step 1 Hz)")
        print("Lags: -250 to +250 ms (step 5 ms)")
        print("Surrogate iterations: 200")
        print("Minimum shift: 2 seconds")
        
        # Optimized parameters
        center_freqs = np.arange(1, 30, dtype=float)  # 1-30 Hz
        lags_ms = np.arange(-250, 251, 5, dtype=float)  # -250 to +250 ms, step 5ms
        n_surrogates = 200  # 200 iterations (balance between speed and precision)
        
        # Analysis for window 1-6 minutes (Morada)
        print("\n" + "="*60)
        print("ANALYSIS WINDOW 1-6 MINUTES (MORADA)")
        print("="*60)
        results_1_6min = run_envelope_correlation_significance_analysis(
            signal_a=signal_pfc,
            signal_b=signal_amy,
            fs=Fs,
            window_name="1_6min",
            start_s=60,      # 1 minute = 60 seconds
            end_s=360,      # 6 minutes = 360 seconds
            center_freqs=center_freqs,
            window_width_hz=2.0,
            lags_ms=lags_ms,
            n_surrogates=n_surrogates,
            min_shift_samples=None
        )
        
        # Analysis for window 11-16 minutes (Laberinto)
        print("\n" + "="*60)
        print("ANALYSIS WINDOW 11-16 MINUTES (LABERINTO)")
        print("="*60)
        results_11_16min = run_envelope_correlation_significance_analysis(
            signal_a=signal_pfc,
            signal_b=signal_amy,
            fs=Fs,
            window_name="11_16min",
            start_s=660,     # 11 minutes = 660 seconds
            end_s=960,      # 16 minutes = 960 seconds
            center_freqs=center_freqs,
            window_width_hz=2.0,
            lags_ms=lags_ms,
            n_surrogates=n_surrogates,
            min_shift_samples=None
        )
        
        print("\n" + "="*60)
        print("[OK] ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nFiles generated for each window:")
        print("• envelope_correlation_with_significance_{window}_matrix.csv")
        print("• envelope_correlation_with_significance_{window}_heatmap.png")
        print("• envelope_correlation_with_significance_{window}_scatter.png")
        print("• envelope_correlation_with_significance_{window}_zscores.png")
        print("• envelope_correlation_with_significance_{window}_summary.csv")
        print("• envelope_correlation_with_significance_{window}_complete.csv")
        
        print("\nFor higher precision analysis, modify parameters:")
        print("• lags_ms: np.arange(-250, 251, 1, dtype=float)  # step 1 ms")
        print("• n_surrogates: 1000  # more iterations")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found. {e}")
        print("Make sure you have 'allChan_1kHz_clean.mat' in the current directory.")
    except Exception as e:
        print(f"[ERROR] during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()