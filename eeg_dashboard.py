import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import butter, filtfilt, hilbert
from scipy import stats
import os
import glob
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EEG Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================

def load_npy_files():
    """Load available .npy files from the current directory."""
    npy_files = glob.glob("*.npy")
    return sorted(npy_files)

def load_csv_files():
    """Load available .csv files from the current directory."""
    csv_files = glob.glob("*.csv")
    return sorted(csv_files)

def load_mat_files():
    """Load available .mat files from the current directory."""
    mat_files = glob.glob("*.mat")
    return sorted(mat_files)

def load_phase_data(filepath):
    """Load phase data from .npy file."""
    try:
        data = np.load(filepath)
        return data
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

def load_csv_data(filepath):
    """Load data from CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

# ==========================================
# SIGNAL PROCESSING FUNCTIONS
# ==========================================

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist
    high = min(nyquist - 0.01, highcut) / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_phase_signal(signal, lowcut, highcut, fs):
    """Filter signal and extract instantaneous phase using Hilbert transform."""
    filtered = bandpass_filter(signal, lowcut, highcut, fs)
    analytic = hilbert(filtered)
    return np.angle(analytic)

def compute_mrl_with_lags(phase1, phase2, lags_samples):
    """Compute MRL for each lag between two phase signals."""
    N = len(phase1)
    mrls = []
    
    for lag in lags_samples:
        if lag < 0:
            s1 = phase1[-lag:]
            s2 = phase2[:N + lag]
        elif lag > 0:
            s1 = phase1[:N - lag]
            s2 = phase2[lag:]
        else:
            s1 = phase1
            s2 = phase2
            
        phase_diff = np.exp(1j * (s1 - s2))
        mrl = np.abs(np.mean(phase_diff))
        mrls.append(mrl)
    
    return np.array(mrls)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def create_mrl_plot(lags_ms, mrls, title="MRL Analysis"):
    """Create interactive Plotly plot for MRL analysis."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lags_ms,
        y=mrls,
        mode='lines+markers',
        name='MRL',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Add vertical line at lag=0
    fig.add_vline(x=0, line_dash="dash", line_color="red", 
                  annotation_text="Lag 0 ms", annotation_position="top")
    
    # Highlight maximum MRL
    max_idx = np.argmax(mrls)
    max_lag = lags_ms[max_idx]
    max_mrl = mrls[max_idx]
    
    fig.add_trace(go.Scatter(
        x=[max_lag],
        y=[max_mrl],
        mode='markers',
        name=f'Max MRL ({max_mrl:.3f})',
        marker=dict(color='red', size=12, symbol='star')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Lag (ms)",
        yaxis_title="Mean Resultant Length (MRL)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_frequency_analysis_plot(frequencies, offsets, title="Frequency Analysis"):
    """Create interactive plot for frequency-based offset analysis."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=offsets,
        mode='lines+markers',
        name='Offset (ms)',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8)
    ))
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="MRL Maximum Offset (ms)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_phase_plot(time, phase1, phase2, title="Phase Comparison"):
    """Create plot comparing two phase signals."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=phase1,
        mode='lines',
        name='Phase 1',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=phase2,
        mode='lines',
        name='Phase 2',
        line=dict(color='#ff7f0e', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Phase (radians)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    st.markdown('<div class="main-header">🧠 EEG Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation and settings
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Analysis Page", [
        "Data Explorer",
        "MRL Analysis", 
        "Frequency Analysis",
        "Phase Comparison",
        "Settings"
    ])
    
    # Common settings in sidebar
    st.sidebar.title("Settings")
    fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=1000, min_value=100, max_value=10000)
    
    # Load available files
    npy_files = load_npy_files()
    csv_files = load_csv_files()
    mat_files = load_mat_files()
    
    if page == "Data Explorer":
        st.markdown('<div class="section-header">📁 Data Explorer</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Signal Files (.npy)")
            if npy_files:
                selected_npy = st.selectbox("Select signal file", npy_files)
                if selected_npy and st.button("Load Signal", key="load_npy"):
                    data = load_phase_data(selected_npy)
                    if data is not None:
                        st.session_state[f'signal_{selected_npy}'] = data
                        st.success(f"Loaded {selected_npy}: {data.shape} samples")
                        
                        # Display basic info
                        st.info(f"""
                        **File Info:**
                        - Shape: {data.shape}
                        - Duration: {len(data)/fs:.2f} seconds
                        - Sample rate: {fs} Hz
                        """)
                        
                        # Plot first few seconds
                        plot_duration = min(5, len(data)/fs)
                        plot_samples = int(plot_duration * fs)
                        time = np.arange(plot_samples) / fs
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=time,
                            y=data[:plot_samples],
                            mode='lines',
                            name='Signal'
                        ))
                        fig.update_layout(
                            title=f"First {plot_duration} seconds of {selected_npy}",
                            xaxis_title="Time (s)",
                            yaxis_title="Amplitude"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No .npy files found in current directory")
        
        with col2:
            st.subheader("Available Analysis Results (.csv)")
            if csv_files:
                selected_csv = st.selectbox("Select CSV file", csv_files)
                if selected_csv and st.button("Load CSV", key="load_csv"):
                    data = load_csv_data(selected_csv)
                    if data is not None:
                        st.session_state[f'csv_{selected_csv}'] = data
                        st.success(f"Loaded {selected_csv}: {data.shape}")
                        st.dataframe(data.head(10))
                        
                        # Basic statistics
                        st.subheader("Data Statistics")
                        st.write(data.describe())
            else:
                st.warning("No .csv files found in current directory")
    
    elif page == "MRL Analysis":
        st.markdown('<div class="section-header">📊 MRL Analysis</div>', unsafe_allow_html=True)
        
        # Check if we have loaded signals
        loaded_signals = {k: v for k, v in st.session_state.items() if k.startswith('signal_')}
        
        if len(loaded_signals) < 2:
            st.warning("Please load at least 2 signal files from the Data Explorer page first")
        else:
            # Select signals for comparison
            signal_names = list(loaded_signals.keys())
            signal1_name = st.selectbox("Select first signal", signal_names, key="mrl_signal1")
            signal2_name = st.selectbox("Select second signal", signal_names, index=1 if len(signal_names) > 1 else 0, key="mrl_signal2")
            
            # Analysis parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                freq_center = st.number_input("Center Frequency (Hz)", value=8, min_value=1, max_value=50)
            with col2:
                bandwidth = st.number_input("Bandwidth (Hz)", value=2, min_value=1, max_value=10)
            with col3:
                max_lag_ms = st.number_input("Max Lag (ms)", value=250, min_value=10, max_value=1000)
            
            # Time window selection
            col1, col2 = st.columns(2)
            with col1:
                start_min = st.number_input("Start Time (minutes)", value=0.0, min_value=0.0)
            with col2:
                duration_min = st.number_input("Duration (minutes)", value=1.0, min_value=0.1, max_value=10.0)
            
            if st.button("Run MRL Analysis", key="run_mrl"):
                signal1 = loaded_signals[signal1_name]
                signal2 = loaded_signals[signal2_name]
                
                # Extract time window
                start_idx = int(start_min * 60 * fs)
                dur_samples = int(duration_min * 60 * fs)
                
                signal1_window = signal1[start_idx:start_idx + dur_samples]
                signal2_window = signal2[start_idx:start_idx + dur_samples]
                
                # Ensure both windows have same length
                min_len = min(len(signal1_window), len(signal2_window))
                signal1_window = signal1_window[:min_len]
                signal2_window = signal2_window[:min_len]
                
                # Compute phases
                f_low = max(0.1, freq_center - bandwidth/2)
                f_high = freq_center + bandwidth/2
                
                with st.spinner("Computing phase signals..."):
                    phase1 = get_phase_signal(signal1_window, f_low, f_high, fs)
                    phase2 = get_phase_signal(signal2_window, f_low, f_high, fs)
                
                # Compute MRL with lags
                max_lag_sec = max_lag_ms / 1000.0
                max_lag_samples = int(max_lag_sec * fs)
                lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
                lags_ms = (lags_samples / fs) * 1000.0
                
                with st.spinner("Computing MRL for different lags..."):
                    mrls = compute_mrl_with_lags(phase1, phase2, lags_samples)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max MRL", f"{np.max(mrls):.4f}")
                with col2:
                    max_idx = np.argmax(mrls)
                    st.metric("Optimal Lag", f"{lags_ms[max_idx]:.2f} ms")
                with col3:
                    st.metric("Frequency Range", f"{f_low:.1f}-{f_high:.1f} Hz")
                
                # Plot results
                fig = create_mrl_plot(lags_ms, mrls, 
                                      title=f"MRL Analysis: {signal1_name} vs {signal2_name}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Store results for download
                results_df = pd.DataFrame({
                    'lag_ms': lags_ms,
                    'mrl': mrls
                })
                st.session_state['mrl_results'] = results_df
    
    elif page == "Frequency Analysis":
        st.markdown('<div class="section-header">📈 Frequency Analysis</div>', unsafe_allow_html=True)
        
        loaded_signals = {k: v for k, v in st.session_state.items() if k.startswith('signal_')}
        
        if len(loaded_signals) < 2:
            st.warning("Please load at least 2 signal files from the Data Explorer page first")
        else:
            signal_names = list(loaded_signals.keys())
            signal1_name = st.selectbox("Select first signal", signal_names, key="freq_signal1")
            signal2_name = st.selectbox("Select second signal", signal_names, index=1 if len(signal_names) > 1 else 0, key="freq_signal2")
            
            # Frequency range parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                freq_min = st.number_input("Min Frequency (Hz)", value=1, min_value=1, max_value=30)
            with col2:
                freq_max = st.number_input("Max Frequency (Hz)", value=30, min_value=1, max_value=50)
            with col3:
                freq_step = st.number_input("Frequency Step (Hz)", value=1, min_value=0.5, max_value=5)
            
            # Lag parameters
            col1, col2 = st.columns(2)
            with col1:
                max_lag_ms = st.number_input("Max Lag (ms)", value=250, min_value=10, max_value=1000, key="freq_max_lag")
            with col2:
                bandwidth = st.number_input("Bandwidth (Hz)", value=2, min_value=1, max_value=10, key="freq_bandwidth")
            
            # Time window
            col1, col2 = st.columns(2)
            with col1:
                start_min = st.number_input("Start Time (minutes)", value=0.0, min_value=0.0, key="freq_start")
            with col2:
                duration_min = st.number_input("Duration (minutes)", value=1.0, min_value=0.1, max_value=10.0, key="freq_duration")
            
            if st.button("Run Frequency Analysis", key="run_freq"):
                signal1 = loaded_signals[signal1_name]
                signal2 = loaded_signals[signal2_name]
                
                # Extract time window
                start_idx = int(start_min * 60 * fs)
                dur_samples = int(duration_min * 60 * fs)
                
                signal1_window = signal1[start_idx:start_idx + dur_samples]
                signal2_window = signal2[start_idx:start_idx + dur_samples]
                
                min_len = min(len(signal1_window), len(signal2_window))
                signal1_window = signal1_window[:min_len]
                signal2_window = signal2_window[:min_len]
                
                # Frequency analysis
                frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
                offsets = []
                
                max_lag_sec = max_lag_ms / 1000.0
                max_lag_samples = int(max_lag_sec * fs)
                lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
                lags_ms = (lags_samples / fs) * 1000.0
                
                progress_bar = st.progress(0)
                
                for i, freq_center in enumerate(frequencies):
                    f_low = max(0.1, freq_center - bandwidth/2)
                    f_high = freq_center + bandwidth/2
                    
                    phase1 = get_phase_signal(signal1_window, f_low, f_high, fs)
                    phase2 = get_phase_signal(signal2_window, f_low, f_high, fs)
                    
                    mrls = compute_mrl_with_lags(phase1, phase2, lags_samples)
                    max_idx = np.argmax(mrls)
                    offsets.append(lags_ms[max_idx])
                    
                    progress_bar.progress((i + 1) / len(frequencies))
                
                # Plot results
                fig = create_frequency_analysis_plot(frequencies, np.array(offsets),
                                                     title=f"Frequency Analysis: {signal1_name} vs {signal2_name}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Store results
                freq_results_df = pd.DataFrame({
                    'frequency_hz': frequencies,
                    'optimal_offset_ms': offsets
                })
                st.session_state['freq_results'] = freq_results_df
    
    elif page == "Phase Comparison":
        st.markdown('<div class="section-header">🌊 Phase Comparison</div>', unsafe_allow_html=True)
        
        loaded_signals = {k: v for k, v in st.session_state.items() if k.startswith('signal_')}
        
        if len(loaded_signals) < 2:
            st.warning("Please load at least 2 signal files from the Data Explorer page first")
        else:
            signal_names = list(loaded_signals.keys())
            signal1_name = st.selectbox("Select first signal", signal_names, key="phase_signal1")
            signal2_name = st.selectbox("Select second signal", signal_names, index=1 if len(signal_names) > 1 else 0, key="phase_signal2")
            
            # Parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                freq_center = st.number_input("Center Frequency (Hz)", value=8, min_value=1, max_value=50, key="phase_freq")
            with col2:
                bandwidth = st.number_input("Bandwidth (Hz)", value=2, min_value=1, max_value=10, key="phase_bw")
            with col3:
                plot_duration = st.number_input("Plot Duration (seconds)", value=2.0, min_value=0.5, max_value=10.0)
            
            # Time window
            col1, col2 = st.columns(2)
            with col1:
                start_min = st.number_input("Start Time (minutes)", value=0.0, min_value=0.0, key="phase_start")
            with col2:
                duration_min = st.number_input("Duration (minutes)", value=1.0, min_value=0.1, max_value=10.0, key="phase_duration")
            
            if st.button("Compare Phases", key="run_phase"):
                signal1 = loaded_signals[signal1_name]
                signal2 = loaded_signals[signal2_name]
                
                # Extract time window
                start_idx = int(start_min * 60 * fs)
                dur_samples = int(duration_min * 60 * fs)
                
                signal1_window = signal1[start_idx:start_idx + dur_samples]
                signal2_window = signal2[start_idx:start_idx + dur_samples]
                
                min_len = min(len(signal1_window), len(signal2_window))
                signal1_window = signal1_window[:min_len]
                signal2_window = signal2_window[:min_len]
                
                # Compute phases
                f_low = max(0.1, freq_center - bandwidth/2)
                f_high = freq_center + bandwidth/2
                
                phase1 = get_phase_signal(signal1_window, f_low, f_high, fs)
                phase2 = get_phase_signal(signal2_window, f_low, f_high, fs)
                
                # Plot only first few seconds
                plot_samples = int(min(plot_duration, len(phase1)/fs) * fs)
                time = np.arange(plot_samples) / fs
                
                fig = create_phase_plot(time, phase1[:plot_samples], phase2[:plot_samples],
                                       title=f"Phase Comparison: {signal1_name} vs {signal2_name}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Phase difference
                phase_diff = phase1[:plot_samples] - phase2[:plot_samples]
                
                fig_diff = go.Figure()
                fig_diff.add_trace(go.Scatter(
                    x=time,
                    y=phase_diff,
                    mode='lines',
                    name='Phase Difference',
                    line=dict(color='#2ca02c', width=1)
                ))
                fig_diff.update_layout(
                    title="Phase Difference",
                    xaxis_title="Time (s)",
                    yaxis_title="Phase Difference (radians)"
                )
                st.plotly_chart(fig_diff, use_container_width=True)
    
    elif page == "Settings":
        st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        st.subheader("Analysis Parameters")
        st.info("Default parameters used across the dashboard:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Signal Processing:**")
            st.write("- Filter order: 3 (Butterworth)")
            st.write("- Filter type: Bandpass")
            st.write("- Phase extraction: Hilbert transform")
        
        with col2:
            st.write("**MRL Analysis:**")
            st.write("- Lag range: ±250 ms (configurable)")
            st.write("- Frequency range: 1-30 Hz (configurable)")
            st.write("- Bandwidth: 2 Hz (configurable)")
        
        st.subheader("Data Management")
        if st.button("Clear All Loaded Data"):
            # Clear all session state except some basic settings
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('signal_') or k.startswith('csv_')]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("All loaded data has been cleared")
        
        st.subheader("Export Results")
        if 'mrl_results' in st.session_state:
            if st.button("Download MRL Results"):
                csv = st.session_state['mrl_results'].to_csv(index=False)
                st.download_button(
                    label="Download MRL Results as CSV",
                    data=csv,
                    file_name='mrl_results.csv',
                    mime='text/csv'
                )
        
        if 'freq_results' in st.session_state:
            if st.button("Download Frequency Results"):
                csv = st.session_state['freq_results'].to_csv(index=False)
                st.download_button(
                    label="Download Frequency Results as CSV",
                    data=csv,
                    file_name='frequency_results.csv',
                    mime='text/csv'
                )
    
    # Footer
    st.markdown("---")
    st.markdown("**EEG Analysis Dashboard** | Interactive MRL and Phase Analysis | Built with Streamlit")

if __name__ == "__main__":
    main()