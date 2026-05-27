# script/process_and_plot_mpdr.py
from pathlib import Path
from cmvdr.util.player import Player
from cmvdr.util import config
import cmvdr.util.globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)
from cmvdr import experiment_manager


def process_files_and_plot_spectrograms(input_files, output_methods=['mvdr_blind', 'cmvdr_blind'],
                                        noise_path=None, save_figs=True):
    """
    Process audio files with MPDR/cMPDR and generate spectrograms.

    Args:
        input_files: List of Path objects or single Path
        output_methods: List of beamforming methods to include in spectrogram
        noise_path: Path to noise folder (optional)
        save_figs: Whether to save figures to demos folder
    """
    # Ensure input_files is a list
    if isinstance(input_files, Path):
        input_files = [input_files]

    # Load configuration
    cfg = config.load_configuration_outer('inference')
    cfg = config.assign_default_values(cfg)
    cfg['beamforming']['methods'] = output_methods

    # Create temporary processing directories
    em = experiment_manager.ExperimentManager()

    # Create input directory symlinks
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix='cmvdr_mpdr_') as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"

        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Create symlinks for input files
        for f in input_files:
            symlink = input_dir / f.name
            if not symlink.exists():
                symlink.symlink_to(f.resolve())

        # Create symlinks for noise files if provided
        noise_dir = None
        if noise_path:
            noise_dir = temp_path / "noise"
            noise_dir.mkdir(exist_ok=True)
            # Match noise files to input files
            from cmvdr.cli.cmvdr_inference_cli import get_audio_files, match_noise_files
            noise_matches = match_noise_files(input_files, Path(noise_path))
            for input_file, noise_file in noise_matches.items():
                symlink = noise_dir / input_file.name
                if not symlink.exists():
                    symlink.symlink_to(noise_file.resolve())

        # Run inference
        em.run_cmvdr_inference_folder(
            input_path=input_dir,
            noise_path=noise_dir,
            output_path=output_dir,
            cfg=cfg,
            verbose=True
        )

        # Load results and prepare for spectrogram plotting
        signals_dict = {}
        fs = 16000

        for output_file in output_dir.rglob('*.wav'):
            import librosa
            y, sr = librosa.load(str(output_file), sr=fs)
            # Extract method name from file path/name
            method_name = output_file.stem.split('_')[-1]  # Adjust based on your naming
            signals_dict[method_name] = y

        # Plot spectrograms
        if signals_dict:
            spectrogram_keys = [m for m in output_methods if m in signals_dict]
            if spectrogram_keys:
                Player.plot_mel_spectrograms_2_by_2(
                    signals_dict,
                    spectrogram_keys,
                    fs=fs,
                    save_figs=save_figs
                )


if __name__ == "__main__":
    # Single file
    input_file = Path("/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/10min_brushless_5s/noisy/noisy_fileid_0_clean_fileid_100-clean_fileid_1003_1850_snr-12_tl-24.wav")
    process_files_and_plot_spectrograms(input_file, output_methods=['mvdr_blind', 'cmvdr_blind'])

    # Multiple files
    # input_files = list(Path("path/to/audio/folder").glob("*.wav"))
    # process_files_and_plot_spectrograms(
    #     input_files,
    #     output_methods=['mpdr_blind', 'cmpdr_blind'],
    #     noise_path="path/to/noise/folder",
    #     save_figs=True
    # )
