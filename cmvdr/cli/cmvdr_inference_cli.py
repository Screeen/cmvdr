import os
from contextlib import redirect_stdout
from pathlib import Path
import argparse
from cmvdr.util import config, globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)
from cmvdr import experiment_manager


def main():
    cfg_original = config.load_configuration_outer('inference')
    cfg_original = config.assign_default_values(cfg_original)
    cfg_original['beamforming']['methods'] = ['cmvdr_blind']

    # Read input path from command line arguments as a positional argument. Use argparse to handle command line arguments.
    parser = argparse.ArgumentParser(description="Run cMVDR inference on a single file or a folder of audio files.")
    parser.add_argument('-i', '--input_path', required=True, type=str, help="Path to the input audio file or folder.")
    parser.add_argument("-o", "--output_path", type=str, default=None,
                        help="Path to the output folder. If not provided, output will be saved in the same folder as input.")
    parser.add_argument("-n", "--noise_path", type=str, default=None,
                        help="Path to the noise audio file or folder (optional, to estimate noise frequency). "
                             "To match input files, append _fileid_123.wav to the noise and the noisy files.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="If set, print detailed logs to the console.")
    args = parser.parse_args()

    em = experiment_manager.ExperimentManager()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    noise_path = Path(args.noise_path).expanduser().resolve() if args.noise_path else None
    kwargs = {'input_path': input_path, 'noise_path': noise_path, 'output_path': output_path,
                'cfg': cfg_original, 'verbose': args.verbose}

    if args.verbose:
        em.run_cmvdr_inference_folder(**kwargs)
    else:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            em.run_cmvdr_inference_folder(**kwargs)


if __name__ == "__main__":
    main()
