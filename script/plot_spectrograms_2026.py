"""
Given a folder, plot mel spectrograms of different algorithms for comparison.
"""

from pathlib import Path
import numpy as np
from matplotlib.ticker import MaxNLocator

from cmvdr.util.player import Player
from cmvdr.util import utils as u
from cmvdr.data_gen.audio_disk_loader import AudioDiskLoader


def get_display_name(x):
    return x.replace('_', ' ')


def plot_mel_spectrograms_2_by_2_dnn(signals, for_spectrogram=None, fs=16000, save_figs=False, fmax=8000):
    """ Plot mel spectrograms in a 2x2 grid. """

    import librosa
    import matplotlib.pyplot as plt
    signals_spec_db = {}

    if for_spectrogram is None:
        for_spectrogram = list(signals.keys())[:4]  # Take first 4 signals if not specified

    # Step 1: Compute mel spectrograms and convert to dB using constant ref
    for key in for_spectrogram:
        mel = librosa.feature.melspectrogram(y=signals[key], sr=fs, n_mels=128, fmax=fmax)
        S_dB = librosa.power_to_db(mel, ref=1000.0)
        signals_spec_db[key] = S_dB

    # Step 2: Global min and max for consistent color range
    min_val = min(np.min(S) for S in signals_spec_db.values())
    max_val = max(np.max(S) for S in signals_spec_db.values())

    fig_size = u.get_plot_width_double_column_latex(), u.get_plot_width_double_column_latex() * 0.85
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=fig_size,
                            sharex=True, sharey=True)

    # Step 3: Plot with fixed color scale
    pcm = None
    for i, (name, S_dB) in enumerate(signals_spec_db.items()):
        row, col = divmod(i, 2)
        pcm = librosa.display.specshow(S_dB, sr=fs, x_axis='time', y_axis='mel', fmax=fmax,
                                       vmin=min_val, vmax=max_val, cmap='magma', ax=axs[row, col])
        # name_disp = get_display_name(name)
        axs[row, col].set_title(name, fontsize=8)

        # Get current tick positions and labels
        tick_locs = axs[row, col].get_yticks()
        tick_labels = [Player.encode_label(val / 1000, pos) for val, pos in zip(tick_locs, tick_locs)]
        axs[row, col].set_yticks(tick_locs, labels=tick_labels)
        axs[row, col].tick_params(axis='x', pad=1)  # smaller pad → labels closer to ticks
        axs[row, col].tick_params(axis='y', pad=1)  # smaller pad → labels closer to ticks
        axs[row, col].set_ylabel("Frequency [kHz]")
        axs[row, col].set_xlabel("Time [s]")

    # Colorbar is shared across all subplots (to the right of the 2x2 grid, as tall as 2 rows)
    cbar = fig.colorbar(format='%+2.0fdB', mappable=pcm, ax=axs, orientation='vertical', fraction=0.06, pad=0.04,
                        aspect=30)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # maximum 4 ticks
    cbar.ax.tick_params(axis='y', pad=1, labelsize=6)  # smaller pad brings labels closer to the bar

    # To avoid repeating xlabels and ylabels in each subplot
    for a in axs.flat:
        a.label_outer()

    fig.show()

    dir_path = Player.get_dir_path()
    name = '_'.join(for_spectrogram)
    name = (name.replace(' ', '_').replace('-', '_').replace('(', '')
            .replace(')', '').replace('+', 'plus').replace('.', '_'))
    if len(name) > 50:
        name = name[:50] + '_etc'

    if save_figs:
        u.savefig(fig, dir_path / f'{name}.pdf')


# Manual renaming. Create a function that takes a dict as input and renames the keys. First, we capitalize first letter.
# then, if contains 'wet' -> clean
# mvdr_blind -> MPDR
# cmvdr_blind -> cMPDR (prop.)
def rename_keys(d):
    new_d = {}
    for k, v in d.items():
        new_k = k.capitalize()
        if 'wet' in new_k.lower():
            new_k = 'Clean'
        elif 'cmvdr_blind' in new_k.lower():
            new_k = 'cMPDR (prop.)'
        elif 'mvdr_blind' in new_k.lower():
            new_k = 'MPDR'
        new_d[new_k] = v
    return new_d


u.set_plot_options(use_tex=True)

# root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h02_real_dregon/audio/noise|snr_db_dir/-20/18_Motor3_90").expanduser()
# root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/noise|snr_db_dir/-20/10_noise-free-sound-0131").expanduser()
root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/noise|snr_db_dir/-20/11_noise-free-sound-0825").expanduser()
root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/noise|snr_db_dir/-20/15_noise-free-sound-0199").expanduser()
root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/M/1/17_noise-free-sound-0438").expanduser()
root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/M/2/1_noise-free-sound-0748").expanduser()
root_path = Path("~/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h16_real_freesound/audio/M/2/8_noise-free-sound-0395").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h45_real_freesound/audio/noise|snr_db_dir/-5/0_noise-free-sound-0010").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h45_real_freesound/audio/noise|snr_db_dir/-5/6_noise-free-sound-0257").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h45_real_freesound/audio/noise|snr_db_dir/-5/5_noise-free-sound-0394").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h59_real_freesound/audio/noise|snr_db_dir/-5/0_noise-free-sound-0257").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h59_real_freesound/"
                 "audio/noise|snr_db_dir/-5/10_noise-free-sound-0257").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/15h59_real_freesound/"
                 "audio/noise|snr_db_dir/-5/18_noise-free-sound-0257").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/16h05_real_freesound/audio/noise|snr_db_dir/-5/12_noise-free-sound-0046").expanduser()

# Good ones
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/16h05_real_freesound/audio/noise|snr_db_dir/-5/40_noise-free-sound-0182").expanduser()
root_path = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/cmvdr/exp_results/2026-02-24/16h05_real_freesound/audio/noise|snr_db_dir/-5/38_noise-free-sound-0002").expanduser()

fs = 16000
audio_files_temp = AudioDiskLoader.load_audio_files(root_path, fs=fs)

# Per each audio file, only keep 'signal' and remove the key
audio_files = {k: v['signal'] for k, v in audio_files_temp.items()}

# Sort them in alphabetical order of keys
audio_files = dict(sorted(audio_files.items()))

# Strip the keys of the extensions
audio_files = {k.rsplit('.', 1)[0]: v for k, v in audio_files.items()}

# Remove first character (used for numbering)
audio_files = {k[2:]: v for k, v in audio_files.items()}

# Keep only first 2 seconds
for k in audio_files.keys():
    audio_files[k] = audio_files[k][:int(fs * 2)]

audio_files = rename_keys(audio_files)

plot_mel_spectrograms_2_by_2_dnn(audio_files, fs=fs, save_figs=True, fmax=2500)
