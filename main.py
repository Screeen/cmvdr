import time
import subprocess
import argparse

from cmvdr.util import config, globs as gs

cfg_original = config.load_configuration('default.yaml')
gs.rng, cfg_original['seed_extracted'] = gs.compute_rng(cfg_original['seed_is_random'],
                                                        cfg_original['seed_if_not_random'])

from cmvdr.experiment_manager import ExperimentManager

# Main entry point for the cyclic beamforming experiment
# This script sets up the experiment, runs it, and manages the output
# Read from command line arguments to configure the experiment
# The script also handles the configuration loading and random number generation setup
if __name__ == '__main__':

    argparse.ArgumentParser(
        description="Run cyclic beamforming experiment with specified configurations."
    )
    parser = argparse.ArgumentParser(description="Cyclic Beamforming Experiment Runner")

    # add command line arguments --data_type with options specified in cfg['data_types_all']
    parser.add_argument(
        '--data_type',
        type=str,
        choices=cfg_original['data_types_all'],
        metavar='DATA_TYPE',  # this will be shown in the help message
        default='config',
        # Show options in the help message
        help=f"Type of data to use for the experiment. Options: {', '.join(cfg_original['data_types_all'])}. Default: 'config'."
    )
    args = parser.parse_args()

    # Load the configuration and merge with secondary config based on data type
    if args.data_type != 'config':
        cfg_original.update({'data_type': args.data_type})
    cfg_original = config.load_and_merge_secondary_config(cfg_original, cfg_original['data_type'])

    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    res = ExperimentManager.run_experiment(cfg_original, args)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s")

    # Open the folder with the figures if the elapsed time is more than 60 seconds or if there are many simulations
    if elapsed_time > 60 or res['cfg_original']['num_montecarlo_simulations'] > 10:
        # Use 'open' command for macOS or 'xdg-open' for Linux
        # To detach, we use stdout and stderr redirection to DEVNULL
        subprocess.Popen(["open", res['target_path_figs']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

