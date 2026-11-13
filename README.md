# live_RPI_viewer

A live viewer for RPI (randomized probe imaging) reconstructions. It reads detector frames from an
EPICS PVA stream, or replays them from an HDF5 file, and reconstructs each frame on the GPU. It has
three backends, and you can chain them together:

- **Fast conjugate gradient**: pure PyTorch (`from_ALS/conjugate_gradient_rpi.py`)
- **Moderate LSQML**: [ptychi](https://github.com/AdvancedPhotonSource/pty-chi)
- **Slow LBFGS**: [cdtools](https://github.com/cdtools-developers/cdtools)

## Requirements

- Linux with an NVIDIA GPU. A CPU-only install starts, but it is too slow for live use.
- Python 3.12 or newer.
- Network access to the detector's PVA server (only for live mode).

## Get the code

```bash
git clone https://github.com/jooneal225/live_RPI_viewer.git
cd live_RPI_viewer
```

## Install with uv (recommended)

The repo includes `pyproject.toml` and a `uv.lock`, so uv rebuilds the exact environment that has
been tested.

```bash
# install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# create .venv/ and install the locked dependencies
uv sync
```

Run the GUI:

```bash
uv run python rpi_live_gui.py
# or activate the env once and use plain python
source .venv/bin/activate
python rpi_live_gui.py
```

To add or upgrade a package, use `uv add <pkg>` or `uv lock --upgrade-package <pkg>`, then commit
the updated `uv.lock`.

## Install with conda

There is no conda environment file. Instead, create an env that has only Python, then use pip to
install the packages listed in `requirements.txt`:

```bash
conda create -n live_rpi python=3.12
conda activate live_rpi
pip install -r requirements.txt
```

Run the GUI:

```bash
conda activate live_rpi
python rpi_live_gui.py
```

This installs everything from PyPI, the same as the uv install, but the versions are not locked.
Avoid mixing in conda packages such as `pytorch` or `pyqt`. The conda builds of these can conflict
with the pip wheels, and `pyqt` from conda is Qt5, but this GUI needs PyQt6.

## GPU / PyTorch notes

On Linux, the default `torch` wheel from PyPI is built for a recent CUDA version (the lock file
currently has `torch 2.15.0+cu132`). If your NVIDIA driver is too old for that version,
`torch.cuda.is_available()` returns `False`. In that case, install a wheel that matches your driver
from the PyTorch index, for example:

```bash
# conda / pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

# uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Note that the next `uv sync` replaces this torch with the locked version.

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`gpu_index` in the `[rpi]` section of `settings.ini` chooses the GPU. The GUI sets
`CUDA_VISIBLE_DEVICES` from it before it imports torch.

## Optional extras

Install these only if you need them:

| Package | Needed for |
| --- | --- |
| `lz4` | PVA streams compressed with the `lz4` codec |
| `bitshuffle` | PVA streams compressed with the `bslz4` codec |
| `blosc2` | PVA streams compressed with the `blosc` codec |

```bash
pip install lz4 bitshuffle blosc2     # conda env
uv pip install lz4 bitshuffle blosc2  # uv env
```

Uncompressed streams and HDF5 replay work without them.

One feature depends on files outside this repo. **quDIS interferometer positions** (in the Object
Accumulator) read `wire.py` and `config.py` from a sibling `../ptychoSAXS/qds_bridge/` checkout. If
those files are missing, that feature turns off and the rest of the GUI works normally.

## Configuration

All settings are stored in `settings.ini`, next to `rpi_live_gui.py`. The GUI rewrites this file
when it exits. Before the first run, edit the paths for your experiment:

- `[paths]`: data directory, probe file, pixel mask, and save directory
- `[epics] pv_name`: the detector's PVA image channel
- `[h5_replay]`: set `enabled = true` and set `h5_file` to replay a saved dataset without a live
  detector

Because the GUI rewrites `settings.ini`, local edits to it show up as changes in git. Commit them
only if you mean to change the defaults.
