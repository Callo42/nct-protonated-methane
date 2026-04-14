# NCT Protonated Methane

Neural Canonical Transformation calculation of vibrational energy levels of
protonated methane (CH5+), implemented in [JAX](https://github.com/jax-ml/jax).

> **Prerequisites — read before installing.** A Fortran compiler is required
> once, to build the JBB CH5+ potential energy surface from source. Installation
> commands per OS are listed under [Install](#install).

## Install

### 1. Install a Fortran compiler

The JBB PES (`J.Phys.Chem.A2006,110,1569-1574`) is shipped as Fortran source and
compiled on your machine via `f2py`, so it matches your Python version and
platform. You only need to do this once.

| OS | Command |
| --- | --- |
| Debian / Ubuntu | `sudo apt install gfortran` |
| Fedora / RHEL   | `sudo dnf install gcc-gfortran` |
| macOS (Homebrew) | `brew install gcc` |
| Windows         | Use [WSL2](https://learn.microsoft.com/windows/wsl/install), then follow the Linux instructions. |

### 2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Sync the Python environment

From the repository root:

```bash
uv sync
```

This creates `.venv/` with Python 3.13 and all locked runtime dependencies.

> By default this installs the **CPU-only** `jaxlib` wheel. NVIDIA GPU users can
> additionally install `jax[cuda12]==0.7.0` into the same venv:
> `uv pip install "jax[cuda12]==0.7.0"`.

### 4. Build the JBB PES extension

```bash
uv run make -C neuralvib/molecule/ch5plus/JBB_Full_PES
```

This produces `JBBCH5ppotential.<abi>.so` next to the Makefile. The extension is
ABI-tied to the Python interpreter inside `.venv/`, so re-run `make` if you ever
recreate the venv.

## Run

Ground-state training (CH5+, JBB PES, RNVP flow) writing checkpoints under
`./data/`:

```bash
uv run python -m neuralvib.train \
    --folder ./data/ \
    --molecule "CH5+" \
    --select_potential "J.Phys.Chem.A2006,110,1569-1574" \
    --num_of_particles 6 \
    --num_orb 1 \
    --flow_type RNVP \
    --flow_depth 16 \
    --mlp_width 128 \
    --mlp_depth 2 \
    --batch 6000 \
    --acc_steps 1 \
    --epoch 20000 \
    --clip_factor 10.0 \
    --optimizer adam \
    --adam_lr 3e-4 \
    --mc_therm 10 \
    --mc_steps 50 \
    --mc_stddev 8.0 \
    --mc_selfadjust_stepsize \
    --excite_gen_type 3
```

Inference from a saved checkpoint:

```bash
uv run python -m neuralvib.inference \
    --file_name <path/to/checkpoint.pkl> \
    --mc_therm 10 --mc_steps 50 --mc_stddev 8.0 --batch 6000
```

## Tests

```bash
uv run pytest neuralvib/molecule/ch5plus/JBB_Full_PES/tests/test_jbb.py
uv run pytest neuralvib
```

## Repository layout

```
neuralvib/
├── train.py                       Training CLI (python -m neuralvib.train)
├── inference.py                   Inference CLI
├── networks/                      RNVP and MoleNet normalizing flows
├── molecule/
│   └── ch5plus/
│       ├── JBB_Full_PES/          Fortran sources + Makefile + JAX wrappers
│       └── McCoy_NN_PES/          Pre-trained Flax neural-network PES
└── utils/                         MCMC, energy estimator, loss, frame, etc.
```

## License

Apache 2.0. See [LICENSE](LICENSE).
