import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_wavelet_transform(data, scales, wavelet_name, fs=256):
    wavelet_transformed_data = []

    for index, row in data.iterrows():
        signal = row[:-1]
        filtered_signal = bandpass_filter(signal, 0.5, 30, fs)
        coefficients, _ = pywt.cwt(filtered_signal, scales, wavelet_name)
        wavelet_transformed_data.append(coefficients)

    return np.array(wavelet_transformed_data)

def remap_labels(y):
    label_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
    return np.vectorize(label_mapping.get)(y)

def calculate_scales():
    sampling_rate = 256
    sampling_interval = 1 / sampling_rate
    central_frequency = pywt.central_frequency('morl')

    # 定义频率范围
    frequency_ranges = [(0.5, 4), (4, 8), (8, 14), (14, 30)]

    scales = []
    for freq_range in frequency_ranges:
        scales_range = [central_frequency / (f * sampling_interval) for f in freq_range]
        scales.append(np.linspace(scales_range[1], scales_range[0], 8))  # 8个尺度
    # 将所有尺度合并为一个数组
    scales = np.hstack(scales)

    return scales

if __name__ == "__main__":
    scales = calculate_scales()
    for person in ['ccq.csv', 'gyf.csv', 'pl.csv', 'tyt.csv', 'whr.csv', 'wl.mat', 'wm.csv']:
        name = person.split('.')[0]
        data = load_data(f'data/{name}.csv')
        transformed_data = apply_wavelet_transform(data, scales, 'morl')
        np.save(f'wavelet/{name}.npy', transformed_data)
        print(f'shape of {name} wavelet_transformed: {transformed_data.shape}')

        # 从原始数据中提取标签
        labels = data.iloc[:, -1].values
        remapped_labels = remap_labels(labels)
        np.save(f'wavelet/{name}_label.npy', remapped_labels)
        print(f'shape of {name} remapped_labels: {remapped_labels.shape}')
