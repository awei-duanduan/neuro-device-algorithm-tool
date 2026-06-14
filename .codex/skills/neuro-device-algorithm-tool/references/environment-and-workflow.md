# Environment And Workflow Reference

See `docs/environment-and-workflow.md` in the project root for the repository version of this guide.

Key files:

- MATLAB GUI: `fit_device_gui.m`
- LTP/LTD fitting: `fit_device_to_realsim.m`
- RealDevice target: `Cell.cpp`
- PyTorch MLP runner: `torch_neurosim_train.py`
- Reservoir runner: `rc_reservoir_train.py`
- Environment checker: `tools/check_env.py`
- First-run setup: `tools/first_run_setup.py`

Default reservoir paper mode uses p-NDI 5-bit/32-state device-response coding with MNIST digits, EMNIST L/M/S, and selected Fashion-MNIST classes.

For new clones, run `python tools/first_run_setup.py` from the project root. It checks the environment and downloads/generates local NeuroSim train/test files from public MNIST/Fashion-MNIST sources.
