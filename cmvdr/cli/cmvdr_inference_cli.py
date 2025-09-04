import os
from contextlib import redirect_stdout
from pathlib import Path
import argparse
from cmvdr.util import config, globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=0, verbose=False)
from cmvdr import experiment_manager

def main():
    cfg_original = config.load_configuration('inference_cmvdr.yaml', verbose=False)
    cfg_original = config.assign_default_values(cfg_original)
    cfg_original['beamforming']['methods'] = ['cmvdr_blind']

    # Read input path from command line arguments as a positional argument. Use argparse to handle command line arguments.
    parser = argparse.ArgumentParser(description="Run cMVDR inference on a single file or a folder of audio files.")
    parser.add_argument('input_path', type=str, help="Path to the input audio file or folder.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="If set, print detailed logs to the console.")

    args = parser.parse_args()

    em = experiment_manager.ExperimentManager()

    input_path = Path(args.input_path)
    if args.verbose:
        em.run_cmvdr_inference_folder(input_path=input_path.expanduser().resolve(),
                                      cfg=cfg_original, output_path=None)
    else:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            em.run_cmvdr_inference_folder(input_path=input_path.expanduser().resolve(),
                                          cfg=cfg_original, output_path=None, verbose=False)

    print(f"Output saved in the same folder as input: {input_path.parent.resolve()}")



if __name__ == "__main__":
    main()
