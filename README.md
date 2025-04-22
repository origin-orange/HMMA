#  Hybrid Mamba-Transformer MARL


## Directory Structure

```bash
root/
├── eval_hmma.py                  #  Main entry point for evaluation (demo using random initialization)
├── networks/
│   └── hmma_networks.py          #  Network definition (Hybrid Mamba-Transformer)
└── mava/
    └── configs/
        └── default_mtm.yaml      #  Hydra configuration (environment, model parameters, etc.)
```

---

##  Environment Setup

### 1. Install Dependencies

Please refer to the [JaxMARL project](https://github.com/oxwhirl/jaxmarl) for environment setup.

Install core dependencies including:

- `jax`
- `flax`
- `chex`
- `optax`

Also install the StarCraft II environments:

- `SMACv2`

### 2. Install Mava Framework

Use InstaDeep's development branch of [Mava](https://github.com/instadeepai/Mava):

```bash
pip install "mava @ git+https://github.com/instadeepai/Mava@develop"
```

> **Note:** The project assumes that either Mava is installed globally or its source code is available under the `mava/` directory of this repository.

### 3. Additional Dependencies

```bash
pip install -r requirements.txt
```

--- The requirements.txt list the main dependencies of the project. 

##  Quick Evaluation

To run a quick test over multiple SMACv2 scenarios, execute:

```bash
python /root/eval_hmma.py
```

## Citing HMMA
If you use HMMA in your work, please cite us as follows:

```
@article{2025hybrid,
  title={Hybrid Mamba-Transformer Multi-Agent Reinforcement Learning for Scalable Coordination in Complex environments},
  author={Kai C, Zhihua C, Lei D, Zhe W, Xin C},
  journal={The Visual Computer},
  year={2025}
}
