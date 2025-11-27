"""
Quick test for loading and processing modulated audio dataset.
"""
import copy
import numpy as np
from pathlib import Path

from cmvdr.beamforming import beamformer_manager
from cmvdr.data_gen.f0_manager import F0Manager
from cmvdr.util import config, globs as gs
from cmvdr.util.harmonic_info import HarmonicInfo

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)

from cmvdr.data_gen.audio_disk_loader import AudioDiskLoader
from cmvdr.data_gen.data_generator import DataGenerator
import cmvdr.util.utils as u


# ------------------------------------------------------------
datasets_path = Path(__file__).parents[2] / 'datasets' / 'dev_datasets_5h' / 'noisy_mod'
# file_name = r'noisy_fileid_45_book_00789_chp_0002_reader_09579_1_noise_000374_snr-5_tl-33_mod.wav'
# file_name = r'noisy_fileid_62_book_02772_chp_0016_reader_09517_26_noise_003674_snr-9_tl-29_mod.wav'
# file_name = r'noisy_fileid_92_book_11110_chp_0045_reader_09003_8_noise_004169_snr-7_tl-17_mod.wav'
file_name = r'noisy_fileid_108_book_09395_chp_0011_reader_07848_14_noise_000521_snr-5_tl-29_mod.wav'
# file_name = r'noisy_fileid_0_book_05924_chp_0008_reader_05011_47_noise_000154_snr3_tl-27_mod.wav'
modulated_file = datasets_path / file_name
assert modulated_file.exists(), f"Modulated file does not exist: {modulated_file}"

frequency_info_path = datasets_path.parent / 'noise' / 'mod_frequencies.pkl'
assert frequency_info_path.exists(), f"Frequency info file does not exist: {frequency_info_path}"
with open(frequency_info_path, 'rb') as f:
    import pickle
    mod_frequencies_dict = pickle.load(f)

# Search through the dict to find 'fileid'. the root keys are paths though, so we need to search
mod_freqs_hz = None
fileid = AudioDiskLoader.find_id_from_filename(file_name)
fun_freq_hz = -1
for key_path, freq_info in mod_frequencies_dict.items():
    if freq_info['fileid'] == fileid:
        mod_freqs_hz = copy.deepcopy(freq_info['mod_freqs'])
        fun_freq_hz = np.abs(mod_freqs_hz[3]) - np.abs(mod_freqs_hz[2]) # fundamental frequency is the second element
        assert 80 < fun_freq_hz < 250, f"Unexpected fundamental frequency: {fun_freq_hz} Hz"
        print(f"Found modulation frequencies for fileid {fileid}: {mod_freqs_hz}, fundamental frequency: {fun_freq_hz} Hz")
        break

# Load clean signal
clean_file_name = 'clean_fileid_' + fileid + '.wav'
clean_path = datasets_path.parent / 'clean' / clean_file_name
assert clean_path.exists(), f"Clean file does not exist: {clean_path}"
clean_audio_dict = AudioDiskLoader.load_audio_files(clean_path, fs=16000)
clean_audio_data = list(clean_audio_dict.values())[0]['signal']
clean_sample_rate = list(clean_audio_dict.values())[0]['sr']
assert clean_sample_rate == 16000, "Sample rate of clean audio should be 16000 Hz"

# Load the modulated audio file
audio_dict = AudioDiskLoader.load_audio_files(modulated_file, fs=16000)
assert len(audio_dict) == 1, "Should load one audio file"
audio_data_interleaved = list(audio_dict.values())[0]['signal']
sample_rate = list(audio_dict.values())[0]['sr']
assert sample_rate == 16000, "Sample rate should be 16000 Hz"

# Check the shape of the loaded audio data. It should have 14 channels: real and imaginary parts for 7 modulations
# first real part is the original signal, first imaginary part is zero
expected_num_channels = 14
assert audio_data_interleaved.ndim == 2, "Audio data should be 2D (samples, channels)"
assert audio_data_interleaved.shape[
           0] == expected_num_channels, f"Audio data should have {expected_num_channels} channels"
print(f"Loaded audio data shape: {audio_data_interleaved.shape}")

# Assemble back the signal from real and imaginary parts
num_samples = audio_data_interleaved.shape[1]
num_modulations = expected_num_channels // 2
P = num_modulations  # Number of modulations
audio_data_cpx = np.zeros((num_modulations, num_samples), dtype=np.complex64)
for i in range(num_modulations):
    real_part = audio_data_interleaved[2 * i, :]
    imag_part = audio_data_interleaved[2 * i + 1, :]
    audio_data_cpx[i, :] = real_part + 1j * imag_part

# Verify that the first modulation is real-valued (imaginary part should be zero)
assert np.allclose(audio_data_cpx[0, :].imag, 0), "First modulation should be real-valued"
print("First modulation is real-valued as expected.")

# Estimate covariance matrices for the modulated signals
# Simplified processing to test correctness of exported modulated data
M = 1  # Number of microphones
num_harmonics_signal = 10  # Number of harmonics in the signal

cfg = {
    'fs': 16000,
    'stft': {
        'nfft': 1024,
        'overlap_fraction': 0.75,
        'win_name': 'hann'
    }
}
dft_props = config.set_stft_properties(cfg['stft'], cfg['fs'])
dg = DataGenerator()
SFT, SFT_real, freqs_hz = dg.get_stft_objects(dft_props)
audio_data_stft = SFT.stft(audio_data_cpx)
audio_data_stft = audio_data_stft[:, :cfg['stft']['nfft'] // 2 + 1, :]  # Keep only non-redundant freq bins

print(f"STFT shape: {audio_data_stft.shape}")  # Should be (num_modulations, num_freq_bins, num_frames)

# To estimate covariance matrices for modulated signals, we would typically do:
_, K_nfft, L2_frames_chunk = audio_data_stft.shape
cov_dict = {
    'noisy_wb': np.zeros((K_nfft, M * P, M * P), dtype=np.complex64),
    'noisy_nb': np.zeros((K_nfft, M, M), dtype=np.complex64)
}

for kk in range(K_nfft):
    slice_frames = slice(0, L2_frames_chunk)
    cov_dict['noisy_wb'][kk] = audio_data_stft[:, kk, slice_frames] @ audio_data_stft[:, kk, slice_frames].conj().T / L2_frames_chunk
    cov_dict['noisy_nb'][kk] = audio_data_stft[0:1, kk, slice_frames] @ audio_data_stft[0:1, kk, slice_frames].conj().T / L2_frames_chunk

fun_freq_bin = np.abs(fun_freq_hz) / SFT_real.delta_f
max_freqs_hz = fun_freq_hz * (num_harmonics_signal + 0.5)
harmonic_freqs_hz = fun_freq_hz * np.arange(1, num_harmonics_signal)

cyclic_bins, _, _, _ = F0Manager.find_harmonic_bins(harmonic_freqs_hz, all_freqs_hz=freqs_hz,
                                                    freq_range_allowed=(0, max_freqs_hz),
                                                    max_relative_dist_from_harmonic=1.5)

P_all = np.ones(K_nfft, dtype=int)
P_all[cyclic_bins] = P
harmonic_sets = np.ones(K_nfft, dtype=int) * -1
harmonic_sets[cyclic_bins] = 1  # only one harmonic set
alpha_mods = [np.array([0]), mod_freqs_hz]  # first set: no modulations, second set: all modulations

bf_man = beamformer_manager.BeamformerManager(beamformers_names=['cmvdr_blind'], sig_shape_k_m=(K_nfft, M))
hi = HarmonicInfo(harmonic_bins=cyclic_bins, harmonic_sets=harmonic_sets, harmonic_freqs_hz=harmonic_freqs_hz, alpha_mods_sets=alpha_mods)
bf_man.harmonic_info = hi

weights, err_flags, cond_num_cov, sv = bf_man.mvdr.compute_cyclic_mvdr_beamformers(cov_dict, 'blind', cyclic_bins,
                                                                                   P_all=P_all)
slice_bf = slice(0, L2_frames_chunk)
name_input_sig = 'noisy_wb'
audio_data_stft_3d = np.zeros((2, M * P, K_nfft, L2_frames_chunk), dtype=np.complex64)
audio_data_stft_3d[0] = audio_data_stft[0]  # non-modulated part
audio_data_stft_3d[1] = audio_data_stft  # modulated parts

weights_dict = {'cmvdr_blind': weights}

bfd = bf_man.beamform_signals(audio_data_stft[:1], audio_data_stft_3d, slice_bf, weights_dict)

u.plot_matrix(audio_data_stft[0, :100, :130], title='Noisy Signal STFT Magnitude (Original, non-modulated)', amp_range=(-250, -50))
u.plot_matrix(bfd['cmvdr_blind'][:100, :130], title='cMVDR Beamformed Signal STFT Magnitude', amp_range=(-250, -50))

# u.plot_matrix(audio_data_stft[0, cyclic_bins, :130], title='Noisy Signal STFT Magnitude (Original, non-modulated)', amp_range=(-250, -50))
# u.plot_matrix(bfd['cmvdr_blind'][cyclic_bins, :130], title='cMVDR Beamformed Signal STFT Magnitude', amp_range=(-250, -50))

bfd_time = SFT_real.istft(bfd['cmvdr_blind'])
if 0:
    u.play(audio_data_interleaved[0], fs=16000)
    u.play(bfd_time.real, fs=16000)
    u.plot([audio_data_interleaved[0], bfd_time.real], subplot_height=2, titles=['Noisy Signal (Original, non-modulated)', 'cMVDR Beamformed Signal'], fs=16000)

import pesq as pypesq  # pip install https://github.com/ludlows/python-pesq/archive/master.zip

pesq_noisy = pypesq.pesq(ref=clean_audio_data, deg=audio_data_interleaved[0], fs=16000, mode='wb')
print(f" PESQ result before cMVDR beamforming: {pesq_noisy:.3f}")

pesq_res = pypesq.pesq(ref=clean_audio_data, deg=bfd_time, fs=16000, mode='wb')
print(f" PESQ result after cMVDR beamforming: {pesq_res:.3f}")
# ------------------------------------------------------------
