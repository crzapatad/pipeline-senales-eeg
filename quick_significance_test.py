import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'C:\\Users\\zcris\\Desktop\\EEG\\EEG2')
from EEG2.leer_mat3_ import (
    load_mat_data,
    run_envelope_correlation_significance_analysis
)

def main():
    print("=== QUICK SIGNIFICANCE TEST ===")
    print("This is a fast test to demonstrate file generation.")
    print()
    
    try:
        # Load data
        print("=== LOADING DATA ===")
        matriz_eeg, Fs, config = load_mat_data("allChan_1kHz_clean.mat")
        
        print(f"[OK] Data loaded: {matriz_eeg.shape}")
        print(f"[OK] Sampling frequency: {Fs} Hz")
        
        # Select channels
        signal_pfc = matriz_eeg[:, 0]  # First channel (PFC)
        signal_amy = matriz_eeg[:, 16]  # Channel 17 (AMY)
        
        print(f"[OK] Channels selected: PFC (channel 1), AMY (channel 17)")
        
        # Very fast test parameters
        print("\n=== FAST TEST CONFIGURATION ===")
        print("Frequencies: 1-5 Hz only (for speed)")
        print("Lags: -50 to +50 ms, step 10 ms (for speed)")
        print("Surrogate iterations: 20 (for speed)")
        
        # Minimal parameters for quick test
        center_freqs = np.arange(1, 6, dtype=float)  # Only 1-5 Hz
        lags_ms = np.arange(-50, 51, 10, dtype=float)  # Only -50 to +50 ms, step 10ms
        n_surrogates = 20  # Only 20 iterations
        
        # Quick test for window 1-2 minutes only
        print("\n" + "="*60)
        print("QUICK TEST: WINDOW 1-2 MINUTES")
        print("="*60)
        results_test = run_envelope_correlation_significance_analysis(
            signal_a=signal_pfc,
            signal_b=signal_amy,
            fs=Fs,
            window_name="test_1_2min",
            start_s=60,      # 1 minute = 60 seconds
            end_s=120,      # 2 minutes = 120 seconds (short window for speed)
            center_freqs=center_freqs,
            window_width_hz=2.0,
            lags_ms=lags_ms,
            n_surrogates=n_surrogates,
            min_shift_samples=None
        )
        
        print("\n" + "="*60)
        print("[OK] QUICK TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nTest files generated:")
        print("• envelope_correlation_with_significance_test_1_2min_matrix.csv")
        print("• envelope_correlation_with_significance_test_1_2min_heatmap.png")
        print("• envelope_correlation_with_significance_test_1_2min_scatter.png")
        print("• envelope_correlation_with_significance_test_1_2min_zscores.png")
        print("• envelope_correlation_with_significance_test_1_2min_summary.csv")
        print("• envelope_correlation_with_significance_test_1_2min_complete.csv")
        
        print("\nFor full analysis, use run_significance_analysis.py with:")
        print("• center_freqs: np.arange(1, 30, dtype=float)")
        print("• lags_ms: np.arange(-250, 251, 5, dtype=float)")
        print("• n_surrogates: 200")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found. {e}")
        print("Make sure you have 'allChan_1kHz_clean.mat' in the current directory.")
    except Exception as e:
        print(f"[ERROR] during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()