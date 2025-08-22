"""
Run cMVDR inference on a given dataset (folder) and save the beamformed output to a specified output folder.
Folders are specified in the configuration file (inference_cmvdr.yaml).
"""
import os
from pathlib import Path
import subprocess
from cmvdr import config
from cmvdr import globs as gs
cfg_original = config.load_configuration('inference_cmvdr.yaml')
gs.rng, cfg_original['seed_extracted'] = gs.compute_rng(cfg_original['seed_is_random'],
                                                        cfg_original['seed_if_not_random'])

from cmvdr.beamformer_manager import BeamformerManager

cfg_original = config.assign_default_values(cfg_original)
cfg_original['beamforming']['methods'] = BeamformerManager.infer_beamforming_methods(cfg_original['beamforming'])

from cmvdr import experiment_manager

data = cfg_original.get('data', {})
input_dir, output_dir = None, None
if 'input_dir' in data:
    input_dir = data['input_dir']
    input_dir = Path(input_dir).expanduser().resolve()
if output_dir := data.get('output_dir'):
    output_dir = Path(output_dir).expanduser().resolve()

em = experiment_manager.ExperimentManager()
em.run_cmvdr_inference_folder(input_folder=input_dir, output_folder=output_dir, cfg=cfg_original,)

# if OS is macOS, open the output folder
if output_dir and output_dir.exists() and output_dir.is_dir():
    # Open the output directory in Finder on macOS
    if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
        subprocess.Popen(["open", output_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
