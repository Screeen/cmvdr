# GitHub Copilot Instructions for cMVDR

## Project Overview

This is the cMVDR (Cyclic Minimum-Variance Distortionless-Response) beamformer project - a Python implementation of an advanced beamforming algorithm for audio signal processing. The project extends the classic MVDR beamformer to exploit both spatial and spectral correlations for better suppression of almost-periodic noise (e.g., engines, fans, musical instruments).

**Key application areas**: Speech enhancement, hearing aids, smart devices, and acoustic scene analysis.

**Research paper**: https://arxiv.org/abs/2510.18391v1

## Development Environment

### Python Version
- **Required**: Python 3.9+ (for compatibility with librosa)
- **Tested on**: Python 3.11
- **OS tested**: macOS 14.4.1 and Ubuntu 24.04.3 LTS

### Dependencies
- Core dependencies are managed in `pyproject.toml`
- Key libraries: librosa, numpy, scipy, matplotlib, sounddevice, pystoi, pesq, pysepm-evo
- Lock file: `uv.lock` (for reproducible builds)

### Installation
```bash
# Standard pip installation (editable mode)
pip install -e .

# Verify installation
python -c 'import cmvdr; print("cMVDR package successfully imported!")'
```

## Project Structure

```
cmvdr/
├── cmvdr/                      # Main package directory
│   ├── beamforming/           # Core beamforming algorithms (MVDR, cyclic MVDR)
│   ├── cli/                   # Command-line interface tools
│   ├── data_gen/              # Synthetic data generation
│   ├── estimation/            # Signal estimation algorithms
│   ├── eval/                  # Evaluation metrics and tools
│   ├── presets/               # Configuration presets
│   └── util/                  # Utility functions
├── tests/                     # Unit and integration tests
├── configs/                   # YAML configuration files
├── script/                    # Helper scripts for running experiments
├── demo/                      # Demo audio files and examples
├── main.py                    # Main experiment runner
└── pyproject.toml            # Package configuration and dependencies
```

## Code Style and Conventions

### General Guidelines
- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate to improve code clarity
- Write descriptive variable names, especially for signal processing variables
- Keep functions focused and modular

### Signal Processing Conventions
- Use standard variable naming for signal processing:
  - `M`: Number of microphones/channels
  - `K`: Number of frequency bins
  - `P`: Number of cyclic shifts
  - `rtf`: Relative transfer function
  - `mvdr`: Minimum variance distortionless response
- Complex-valued arrays should use `dtype=np.complex128`
- FFT/STFT operations should preserve proper scaling and normalization

### Documentation
- Include docstrings for all public functions and classes
- Document parameters, return values, and any important implementation details
- Reference academic papers or algorithms where applicable

## Testing

### Running Tests
```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_cyclic_mvdr

# Run with verbose output
python -m unittest discover -s tests -v
```

### Test Structure
- Tests are located in the `tests/` directory
- Test files follow the naming convention `test_*.py`
- Use `unittest` framework for all tests
- Tests should be fast and focused on specific functionality
- Integration tests should use small data sizes to keep tests fast

### Writing Tests
- Set up test fixtures in `setUp()` method
- Use small array sizes (e.g., M=2, K=5) for fast execution
- Test edge cases (single channel, zero inputs, boundary conditions)
- Verify shapes and dtypes of outputs
- Use `np.allclose()` for floating-point comparisons

## Command-Line Tools

The package provides two CLI tools:

### cmvdr
Runs beamforming inference on audio files:
```bash
cmvdr -i INPUT_PATH [-o OUTPUT_PATH] [-n NOISE_PATH] [-v]
```

### cmvdr-eval
Evaluates audio quality metrics:
```bash
cmvdr-eval -d FOLDER_DENOISED [-r FOLDER_REFERENCE] [--sort-by-snr]
```

## Running Experiments

### Configuration
- Experiment parameters are in YAML files in `configs/` directory
- Main configuration files: `cmvdr.yaml`, `default.yaml`
- Inference configuration: `inference_cmvdr.yaml`

### Experiment Scripts
```bash
# Quick demo
source script/run_demo.sh

# Synthetic data experiments
source script/run_synthetic.sh

# All experiments
source script/run_all.sh

# Direct execution
python main.py --data_type synthetic  # or instruments
```

## Development Workflow

1. **Make changes**: Edit code in the appropriate module
2. **Run tests**: Verify changes don't break existing functionality
3. **Test locally**: Use demo scripts or CLI tools to validate
4. **Document**: Update docstrings and README if needed
5. **Commit**: Write clear, descriptive commit messages

## Common Patterns

### Working with Audio Signals
- Use librosa for audio I/O: `librosa.load()`, `sf.write()`
- STFT parameters are typically configured in preset files
- Maintain proper audio sample rates (typically 16 kHz or 48 kHz)

### Beamforming Pipeline
1. Load multichannel audio
2. Compute STFT
3. Estimate noise covariance
4. Compute beamformer weights
5. Apply beamformer
6. Inverse STFT
7. Save output

### Configuration Management
- Use YAML files for configuration
- Load configs with `yaml.safe_load()`
- Validate configuration parameters before use

## Important Notes

### Known Issues
- Some systems may require PortAudio library for audio playback
- NumPy version compatibility: Install PESQ from GitHub if needed
- SciPy kaiser import: May need adjustment in pysepm for newer scipy versions

### Performance Considerations
- Beamforming is computationally intensive
- Use appropriate FFT sizes for memory/speed tradeoff
- Consider using smaller data for testing/debugging

### Dataset Requirements
For reproducing paper experiments:
- Room impulse responses (RIRs) from RWTH Aachen database
- Clean speech from Speech Intelligibility CD
- Instrument samples from University of Iowa MIS database

## Contact

For questions and feedback: G.Bologni@tudelft.nl
