"""
Test new ShortTimeFFT class. Test if the STFT and ISTFT are consistent.
"""

# Build long random signal
import numpy as np
from scipy.signal import chirp
from scipy.signal import ShortTimeFFT
import scipy.signal

fs = 2000
x_len = 3000
# x = np.random.randn(x_len)
x = chirp(np.linspace(0, 1, x_len), f0=10, f1=100, t1=1, method='quadratic')
# x = np.sin(2 * np.pi * 100 * np.linspace(0, 1, x_len))
# x_r = np.zeros_like(x)
t = np.arange(len(x)) / fs

num_chunks = 3
x1 = x[:int(x_len / 3)]
x2 = x[int(x_len / 3):int(2 * x_len / 3)]
x3 = x[int(2 * x_len / 3):]

nfft = 512
hop = nfft // 4
nov = nfft - hop
stft = ShortTimeFFT(fs=fs, hop=hop, mfft=nfft, win=scipy.signal.windows.hann(nfft))

X = stft.stft(x)
num_frames = X.shape[-1]
num_frames_per_chunk = num_frames // num_chunks

chunk_len_samples = num_frames_per_chunk * hop + nov
non_overlapping_len_samples = chunk_len_samples - nov
Z = np.zeros((nfft // 2 + 1, nfft // hop - 1), dtype=np.complex128)
x1_r = stft.istft(np.hstack((X[:, :num_frames//3], Z)), k0=0, k1=non_overlapping_len_samples)
x2_r = stft.istft(np.hstack((X[:, num_frames//3:2*num_frames//3], Z)), k0=-nov, k1=non_overlapping_len_samples)
x3_r = stft.istft(np.hstack((X[:, 2*num_frames//3:], Z)), k0=-nov, k1=non_overlapping_len_samples)

# Let's try to reconstruct the signal using the overlap
x_r = np.zeros_like(x)

x_r[:non_overlapping_len_samples] = x1_r

counter = non_overlapping_len_samples
remaining_len = min(len(x_r) - counter, non_overlapping_len_samples)
x_r[counter - nov:counter] += x2_r[:nov]
x_r[counter:counter+remaining_len] = x2_r[nov:nov+remaining_len]

counter = counter + remaining_len
remaining_len = min(len(x_r) - counter, non_overlapping_len_samples)
x_r[counter - nov:counter] += x3_r[:nov]
x_r[counter:counter+remaining_len] = x3_r[nov:nov+remaining_len]

# u.plot(x_r, time_axis=False)

if not np.allclose(x, x_r):
    raise ValueError('Reconstructed signal is not the same as the original signal.')
else:
    pass
    # print('Reconstructed signal is the same as the original signal.')