import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal

def process_wearable_data(raw_data: pd.DataFrame) -> np.ndarray:
    features = {
        'acc': ['x', 'y', 'z'],
        'eda': ['conductance'],
        'hr': ['bpm'],
        'temp': ['celsius']
    }
    window_size = 60 * 30  # 30-minute windows
    overlap = 0.5
    processed_windows = []
    for start in range(0, len(raw_data), int(window_size * overlap)):
        window = raw_data.iloc[start:start + window_size]
        if len(window) == window_size:
            window_features = extract_features(window, features)
            processed_windows.append(window_features)
    return np.array(processed_windows)

def extract_features(window: pd.DataFrame, feature_config: dict) -> np.ndarray:
    features = []
    for signal_type, channels in feature_config.items():
        for channel in channels:
            data = window[f'{signal_type}_{channel}']
            features.extend([np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data)])
            if signal_type in ['acc', 'eda']:
                freqs, psd = signal.welch(data)
                features.extend([np.sum(psd), np.mean(psd), freqs[np.argmax(psd)]])
    return np.array(features)
