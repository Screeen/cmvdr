# cMVDR: Cyclic minimum-variance distortionless-response beamformer
Code accompanying the paper "cMVDR: Cyclic minimum-variance distortionless-response beamformer", 
to be submitted for publication.

<div align="center">
  <img src="pics/detailed_scheme.png" alt="Overview of results on synthetic data" width="500"/>
</div>

The paper proposes a new beamforming method leveraging the cyclostationarity of noise signals.  
By exploiting correlations across microphones and frequency components, the cyclic minimum-variance 
distortionless-response (cMVDR) beamformer achieves improved noise reduction, 
especially in low signal-to-noise ratio (SNR) scenarios.

---

## Installation

### Prerequisites
- Python 3.9 (for compatibility with librosa)
- Tested on macOS 14.4.1 but should run on most Linux systems.

### Setup
1. Clone the repository:
```bash
   git clone git@github.com:Screeen/cMVDR.git
   cd cMVDR
```

2. Create and activate a Python virtual environment (assume `uv` and `python3.9` are installed):
```bash
   uv venv -p python3.9 .venv
   source .venv/bin/activate
```

3. Install required packages:
```bash
   pip install --upgrade pip
   uv sync  # requires the `uv` package for managing virtual environments
```

---

## Testing the installation
### Running the tests
To verify the installation, run the provided tests:
```bash
python -m unittest discover -s tests
```

### Running the quick demo
To quickly test the cMVDR implementation, run the demo script:
```bash
source script/run_demo.sh
```

## Reproducing paper experiments

### Configurations
Experiment parameters are controlled via YAML files in the `configs/` folder. Edit `cmvdr.yaml` and `default.yaml` 
to set your desired parameters such as:
- `data_type` (choose from `synthetic` or `instruments`)
- `num_montecarlo_simulations`
- ... 

#### Run synthetic data experiments:
  ```bash
  source script/run_synthetic.sh
  ```

#### Run all experiments (synthetic and instruments):
```bash
source script/run_all.sh
```

## Get cMVDR output for your own audio files (inference)
To apply the cMVDR beamformer to your own audio files, you can use the inference script.
This script processes audio files in a specified folder and saves the output to another folder.
```bash
python script/cmvdr_inference.py
```
Specify the folder containing audio files in the `config/cmvdr_inference.yaml` YAML file:
```yaml
data:
  input_dir: ../datasets/test_cmvdr/noisy
  output_dir: ../datasets/test_cmvdr/noisy_output
```

### Notes
Synthetic data experiments require downloading room impulse responses (RIRs) and clean speech:
- RIRs: https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/
- Speech: Speech Intelligibility CD by Neil Thompson Shade

Instrument data experiments also require downloading instrument samples:
- Instrument samples:  https://theremin.music.uiowa.edu/MIS.html.

## Troubleshooting

### Cannot import `kaiser` from `scipy.signal`
If you get
```
ImportError: cannot import name 'kaiser' from 'scipy.signal' (.../.venv/lib/python3.X/site-packages/scipy/signal/__init__.py). Did you mean: 'kaiserord'?
```
Open the file
```
nano .../SVD-direct/env/lib/python3.X/site-packages/pysepm/util.py
```
and replace 
```
from scipy.signal import firls,kaiser,upfirdn
```
with
```
from scipy.signal import firls,upfirdn
from scipy.signal.windows import kaiser
```

### UV is not installed
`pip` can install directly from the `pyproject.toml`:
```bash
python3 -m venv .venv
pip install --upgrade pip
pip install .
```

### OSError: PortAudio library not found
Your Python package (`sounddevice` or something else that uses PortAudio) can’t find the underlying PortAudio C library on your system.
Installing the Python package alone isn’t enough — the native library must also be installed.
How you fix it depends on your OS. You can still run the experiments, but you won't be able to listen to the audio output.

### ImportError: numpy.core.multiarray
```
ImportError: numpy.core.multiarray failed to import (auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy; use '<void>numpy._import_array' to disable if you are certain you don't need it).
-bash: read: `REPLY?Press enter to exit': not a valid identifier
```
This error can occur if you have an incompatible version of NumPy installed.
Run
```bash
pip install https://github.com/ludlows/python-pesq/archive/master.zip
```

---

> [!NOTE]
> Feedback and questions welcome: G.Bologni@tudelft.nl.
> Enjoy experimenting with cMVDR!
> 
