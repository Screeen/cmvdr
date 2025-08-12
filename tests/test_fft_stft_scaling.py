import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import get_window, stft
import scipy as sp
import scipy.signal

"""
if scaling == 'density':
   scale = 1.0 / (fs * (win*win).sum())
elif scaling == 'spectrum':
   scale = 1.0 / win.sum()**2
else:
   raise ValueError('Unknown scaling: %r' % scaling)

if mode == 'stft':
   scale = np.sqrt(scale)
"""

# Generate an example speech signal (sum of sine waves)
fs = 16000.  # Sampling rate
t = np.arange(0, 1, 1. / fs)  # Time vector
freq = 1000  # Sine wave frequency of 440Hz
signal = np.sin(2 * np.pi * freq * t)

# Define frame length and hop length
FrameLen = 1024
HopLen = 1 * FrameLen // 8

# Get sqrt Hann window function
win_stft = get_window('hann', FrameLen)
if not scipy.signal.check_COLA(win_stft, FrameLen, FrameLen - HopLen):
    raise ValueError('The window does not satisfy the COLA condition')

win_fft = np.sqrt(win_stft)

# Compute window normalization factor for PSD
scale = 1.0 / win_fft.sum()

# Frame processing
num_frames = (len(signal) - FrameLen) // HopLen + 1

frames = []
for i in range(num_frames):
    start_idx = i * HopLen
    end_idx = start_idx + FrameLen
    frame = signal[start_idx: end_idx]
    frames.append(frame)

# Initialize reconstructed signal and power lists
reconstructed_signal = np.zeros(len(signal))
frame_pow_sum_list = []  # Store power of each frame (FFT processing)

# Windowed frames and PSD calculation
for i, frame in enumerate(frames):
    windowed_frame = frame * win_fft
    fft_frame = sp.fft.rfft(windowed_frame) * scale

    # Calculate PSD of the current frame (FFT processing)
    frame_pow_per_freq = np.abs(fft_frame)
    frame_pow_sum = np.sum(frame_pow_per_freq)  # Integrate over frequency
    frame_pow_sum_list.append(frame_pow_sum)

    ifft_frame = sp.fft.irfft(fft_frame) / (scale / (2 * HopLen / FrameLen))

    # Multiply by window again (synthesis window)
    processed_frame = ifft_frame * win_fft

    # Reconstruct signal using overlap-add
    start_idx = i * HopLen
    reconstructed_signal[start_idx:start_idx + FrameLen] += processed_frame

# Adjust signal length (since the last frame might exceed the range)
reconstructed_signal = reconstructed_signal[:len(signal)]

# Compute PSD using scipy.signal.stft with sqrt Hann window
SFT = scipy.signal.ShortTimeFFT(fs=fs, hop=HopLen, win=win_stft, scale_to='magnitude')
Zxx = SFT.stft(signal)
signal_stft = SFT.istft(Zxx, k1=len(signal))
t_stft = SFT.t(n=len(signal))
f_stft = SFT.f
# f_stft, t_stft, Zxx = stft(signal, fs=fs, window=win, nperseg=FrameLen, noverlap=FrameLen - HopLen, boundary=None)

# Calculate PSD for each frame (STFT processing)
Zxx_original = copy.deepcopy(Zxx)
Zxx = Zxx[:, SFT.lower_border_end[-1]:SFT.upper_border_begin(len(signal))[-1]]
t_stft = t_stft[SFT.lower_border_end[-1]:SFT.upper_border_begin(len(signal))[-1]]
frame_pow = np.abs(Zxx)

# Calculate power of each frame (integrate over frequency)
frame_pow_sum_stft = np.sum(frame_pow, axis=0)

# # Plot original signal and reconstructed signal
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, reconstructed_signal, linestyle='--', label='Reconstructed Signal (FFT)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Original vs Reconstructed Signal')
# plt.show()

# Calculate the error between the reconstructed signal and the original signal
# print(f'Reconstruction Error (FFT processing): {np.mean((signal - reconstructed_signal) ** 2):.10f}')
# print(f"Reconstruction Error (STFT processing): {np.mean((signal - signal_stft) ** 2):.10f}")

# Plot the power of each frame (FFT processing and STFT processing)
plt.figure(figsize=(12, 6))
time_frames = np.arange(num_frames) * HopLen / fs  # Time corresponding to each frame
plt.plot(time_frames, frame_pow_sum_list, label='Frame Power (FFT Processing)')
plt.plot(t_stft, frame_pow_sum_stft, linestyle='--', label='Frame Power (STFT Processing)')
plt.xlabel('Time [s]')
plt.ylabel('Power')
plt.legend()
plt.title('Comparison of Frame Powers between FFT and STFT Processing')
# plt.show()

# Compare the power differences between the two methods
min_len = min(len(frame_pow_sum_list), len(frame_pow_sum_stft))
power_difference = np.abs(np.array(frame_pow_sum_list[:min_len]) - frame_pow_sum_stft[:min_len])
mean_power_difference = np.mean(power_difference)
# print(f'Mean Power Difference between FFT and STFT: {mean_power_difference:.10f}')
