import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import wfdb

# Reading Data
normal_record = ['100', '101']
normal_record_path = [r'normal/100/100', r'normal/101/101'] 

abnormal_record = ['200', '201']
abnormal_record_path = [r'abnormal/200/200', r'abnormal/201/201']

normal_record_data = [wfdb.rdrecord(f'{_}') for _ in normal_record_path]
abnormal_record_data = [wfdb.rdrecord(f'{_}') for _ in abnormal_record_path]

ecg_signal_100 = normal_record_data[0].p_signal[:, 0]
ecg_signal_101 = normal_record_data[1].p_signal[:, 0]

ecg_signal_200 = abnormal_record_data[0].p_signal[:, 0]
ecg_signal_201 = abnormal_record_data[1].p_signal[:, 0]

# Constants
fs = 360
nperseg=1024
noverlap = nperseg // 2
LF_band = (0.04, 0.36)
HF_band = (0.36, 0.8)
threshold = 1.5

# Applying Welch
frequencies_100, psd_100 = welch(ecg_signal_100, fs, nperseg=nperseg, noverlap=nperseg//2)
frequencies_101, psd_101 = welch(ecg_signal_101, fs, nperseg=nperseg, noverlap=nperseg//2)
frequencies_200, psd_200 = welch(ecg_signal_200, fs, nperseg=nperseg, noverlap=nperseg//2)
frequencies_201, psd_201 = welch(ecg_signal_201, fs, nperseg=nperseg, noverlap=nperseg//2)

# Filtering the lower and higher frequencies indices
LF_indices_100 = np.where((frequencies_100 >= LF_band[0]) & (frequencies_100 <= LF_band[1]))
HF_indices_100 = np.where((frequencies_100 >= HF_band[0]) & (frequencies_100 <= HF_band[1]))

LF_indices_101 = np.where((frequencies_101 >= LF_band[0]) & (frequencies_101 <= LF_band[1]))
HF_indices_101 = np.where((frequencies_101 >= HF_band[0]) & (frequencies_101 <= HF_band[1]))

LF_indices_200 = np.where((frequencies_200 >= LF_band[0]) & (frequencies_200 <= LF_band[1]))
HF_indices_200 = np.where((frequencies_200 >= HF_band[0]) & (frequencies_200 <= HF_band[1]))

LF_indices_201 = np.where((frequencies_201 >= LF_band[0]) & (frequencies_201 <= LF_band[1]))
HF_indices_201 = np.where((frequencies_201 >= HF_band[0]) & (frequencies_201 <= HF_band[1]))

# Filtering psd based on above indices
LF_power_100 = np.sum(psd_100[LF_indices_100])
HF_power_100 = np.sum(psd_100[HF_indices_100])

LF_power_101 = np.sum(psd_101[LF_indices_101])
HF_power_101 = np.sum(psd_101[HF_indices_101])

LF_power_200 = np.sum(psd_200[LF_indices_200])
HF_power_200 = np.sum(psd_200[HF_indices_200])

LF_power_201 = np.sum(psd_201[LF_indices_201])
HF_power_201 = np.sum(psd_201[HF_indices_201])

# Calculating lf/hf ratio
lf_hf_ratio_100 = LF_power_100 / HF_power_100
lf_hf_ratio_101 = LF_power_101 / HF_power_101
lf_hf_ratio_200 = LF_power_200 / HF_power_200
lf_hf_ratio_201 = LF_power_201 / HF_power_201

# Preliminary Results
print("100 (Healthy Heart): ")
print(f"LF Power: {LF_power_100}")
print(f"HF Power: {HF_power_100}")
print(f"LF/HF Ratio: {lf_hf_ratio_100}")
if lf_hf_ratio_100 < threshold:
    print("Abnormality detected: High LF/HF ratio!")
else:
    print("Normal signal detected.")

print("101 (Healthy Heart): ")
print(f"LF Power: {LF_power_101}")
print(f"HF Power: {HF_power_101}")
print(f"LF/HF Ratio: {lf_hf_ratio_101}")
if lf_hf_ratio_101 < threshold:
    print("Abnormality detected: High LF/HF ratio!")
else:
    print("Normal signal detected.")

print("200 (Arrythmia): ")
print(f"LF Power: {LF_power_200}")
print(f"HF Power: {HF_power_200}")
print(f"LF/HF Ratio: {lf_hf_ratio_200}")
if lf_hf_ratio_200 < threshold:
    print("Abnormality detected: High LF/HF ratio!")
else:
    print("Normal signal detected.")

print("201 (Weak Heart): ")
print(f"LF Power: {LF_power_201}")
print(f"HF Power: {HF_power_201}")
print(f"LF/HF Ratio: {lf_hf_ratio_201}")
if lf_hf_ratio_201 < threshold:
    print("Abnormality detected: High LF/HF ratio!")
else:
    print("Normal signal detected.")

# Visualizing Frequency vs. Power Spectral Density (PSD) for all datasets
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].semilogy(frequencies_100, psd_100)
axs[0, 0].set_title('Dataset - 100 (Healthy Heart)')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_ylabel('Power Spectral Density (dB/Hz)')

axs[0, 1].semilogy(frequencies_200, psd_200)
axs[0, 1].set_title('Dataset - 200 (Arrythmia)')
axs[0, 1].set_xlabel('Frequency (Hz)')
axs[0, 1].set_ylabel('Power Spectral Density (dB/Hz)')

axs[1, 0].semilogy(frequencies_101, psd_101)
axs[1, 0].set_title('Dataset - 101 (Healthy Heart)')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Power Spectral Density (dB/Hz)')

axs[1, 1].semilogy(frequencies_201, psd_201)
axs[1, 1].set_title('Dataset - 201 (Weak Heart)')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Power Spectral Density (dB/Hz)')

fig.suptitle('Power Spectral Density for Different Datasets', fontsize=16)
plt.tight_layout()
plt.show()

# Comparison of Power Spectral Density: Healthy Heart vs. Heart with Arrythmia
plt.figure(figsize=(10, 6))
plt.title('Dataset-1 (indicating Arrythmia)')
plt.semilogy(frequencies_100, psd_100, color='green', label='Normal')
plt.semilogy(frequencies_200, psd_200, color='red', label='Abnormal')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.xlabel('Frequency (Hz)')
plt.legend() 
plt.show()

# Comparison of Power Spectral Density: Healthy Heart vs. Weak Heart
plt.figure(figsize=(10, 6))
plt.title('Dataset-2 (indicating Weak Heart)')
plt.semilogy(frequencies_101, psd_101, color='green', label='Normal')
plt.semilogy(frequencies_201, psd_201, color='red', label='Abnormal')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.xlabel('Frequency (Hz)')
plt.legend() 
plt.show()
