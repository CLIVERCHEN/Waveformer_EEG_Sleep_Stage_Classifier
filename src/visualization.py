import matplotlib.pyplot as plt
import numpy as np
import pywt
import random

def calculate_scales():
    sampling_rate = 256
    sampling_interval = 1 / sampling_rate
    central_frequency = pywt.central_frequency('morl')

    frequency_ranges = [(0.5, 4), (4, 8), (8, 14), (14, 30)]

    scales = []
    for freq_range in frequency_ranges:
        scales_range = [central_frequency / (f * sampling_interval) for f in freq_range]
        scales.append(np.linspace(scales_range[1], scales_range[0], 8))  # 8个尺度
    scales = np.hstack(scales)
    return scales

if __name__ == "__main__":

    transformed_data = np.load('wavelet_transformed_data.npy')
    remapped_labels = np.load('labels_data.npy')

    scales = calculate_scales()

    # 随机选择10个小波变换结果进行可视化
    num_samples_to_visualize = 10
    selected_indices = random.sample(range(transformed_data.shape[0]), num_samples_to_visualize)

    # 创建大图
    fig = plt.figure(figsize=(20, 10))

    for i, idx in enumerate(selected_indices):
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')

        coef = transformed_data[idx]

        X, Y = np.meshgrid(range(coef.shape[1]), scales)

        # 绘制小波变换结果的三维曲面图
        surf = ax.plot_surface(X, Y, coef, cmap='viridis', edgecolor='none')

        ax.set_title(f'Class: {remapped_labels[idx]}')
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
        ax.set_zlabel('Coefficient')

    plt.tight_layout()
    plt.show()
