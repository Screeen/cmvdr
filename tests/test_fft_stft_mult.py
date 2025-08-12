# Verify multiplication/convolution property with mixed FFT and STFT processing
import numpy as np
import scipy.signal
from src import utils as u

N = 4000
Nh = 600
nfft = 1024
hop = 256
fs = 4000

# Sum 5 cosine waves with different frequencies
norm_freqs = np.random.uniform(size=5) * 1000
x = np.sum(np.cos(2 * np.pi * norm_freqs[:, None] * np.arange(N) / N), axis=0)
h = np.random.randn(Nh) / 10

win = scipy.signal.get_window('hann', nfft, fftbins=True)
STF = scipy.signal.ShortTimeFFT(win, hop=hop, fs=fs, fft_mode='twosided', scale_to='magnitude')

y1 = np.convolve(x, h[:nfft], mode='full')[:N].real
Y1 = STF.stft(y1)

X = STF.stft(x)
h_pad = np.zeros(nfft)
h_pad[:Nh] = h
H = scipy.fft.fft(h_pad, nfft)
Y2 = X * H[:, None]
y2 = STF.istft(Y2)[:N].real

# u.plot([y1, y2], ['conv', 'fft'])



