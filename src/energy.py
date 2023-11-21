import numpy as np

def calculate_wavelet_energy(coefficients):
    #squared_coefficients = coefficients ** 2
    squared_coefficients = np.log(np.abs(coefficients)+1)
    energy = np.sum(squared_coefficients, axis=1)
    return energy


def generate_energy_npy(original_npy_path, output_npy_path):
    # 加载原始.npy文件
    wavelet_data = np.load(original_npy_path)

    # 计算每个小波变换结果的能量
    energy_data = []
    for coefficients in wavelet_data:
        energy = calculate_wavelet_energy(coefficients)
        energy_data.append(energy)

    #print(f'归一化前均值：{np.mean(energy_data)}')
    #print(f'归一化前标准差：{np.std(energy_data)}')
    min_val = np.min(energy_data)
    #print(f'最小值：{min_val}')
    max_val = np.max(energy_data)
    #print(f'最大值：{max_val}')
    energy_data = (energy_data - min_val) / (max_val - min_val)
    #print(f'归一化后均值：{np.mean(energy_data)}')
    #print(f'归一化后标准差：{np.std(energy_data)}')

    np.save(output_npy_path, np.array(energy_data))
    return energy_data.shape

if __name__ == "__main__":
    for name in ['ccq', 'gyf', 'pl', 'tyt', 'whr', 'wl', 'wm']:
        shape = generate_energy_npy(f'wavelet/{name}.npy', f'energy/{name}.npy')
        print(f'shape of {name} energy: {shape}')
