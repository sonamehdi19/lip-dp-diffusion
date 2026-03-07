# Lip-DP-Diffusion

PyTorch implementation of **Lip-DP-Diffusion**, a differentially private 
training framework for Denoising Diffusion Probabilistic Models (DDPMs) that 
replaces per-sample gradient clipping with rigorous architectural Lipschitz 
constraints.

Rather than applying DP-SGD's heuristic gradient clipping, Lip-DP-Diffusion 
enforces spectral norm bounds on all trainable layers, enabling 
**data-independent, per-layer sensitivity computation** and calibrated noise 
injection without materialising per-sample gradients. This results in 
unbiased gradient estimates and significant training speedups over DP-SGD.

## Contents

| File | Description |
|---|---|
| `Lip-DP-Diffusion.ipynb` | Core training pipeline: Lip-DP-Diffusion from scratch on MNIST / Fashion-MNIST |
| `Privacy Audit.ipynb` | Empirical privacy auditing via Loss-based Membership Inference Attack and outlier memorisation analysis |
| `fine_tune.ipynb` | Transfer learning with Lip-DP-Diffusion: public pre-training + private fine-tuning |
| `dp_sgd_baseline.ipynb` | DP-SGD baseline with noise multiplicity, EMA, and classifier-free guidance |

## Key Features

- **Per-layer noise calibration** via forward input bound propagation and 
  backward gradient Lipschitz tracking across the U-Net DAG
- **Spectral norm clipping** with inequality constraints: layers with 
  $\sigma(W) < C$ are left untouched, injecting less noise and preserving utility
- **Bounded FiLM conditioning** for timestep and class conditioning, ensuring 
  non-expansive scale factors for sound sensitivity tracking
- **RDP privacy accounting** via Rényi Differential Privacy with Poisson 
  subsampling amplification
- **Transfer learning protocol**: public pre-training within the constrained 
  weight space followed by private fine-tuning on sensitive data
- Evaluated on **MNIST** and **Fashion-MNIST** with FID/IS for data quality and classifier accuracy for downstream utility with Train-on-Synthetic, Test-on-Real (TSTR) protocol

## Method

Lip-DP-Diffusion builds on the weight-clipping framework of 
[Barczewski & Ramon (2025)](https://arxiv.org/abs/...) and extends it to the 
multi-scale DAG topology of conditional U-Nets, handling skip connections, 
scaled concatenations, residual additions, and FiLM conditioning in a 
principled way.

## Citation

If you use this code in your research, please cite:
```bibtex
@mastersthesis{mehdizade2026lipdp,
  title     = {Exploration of Differential Privacy in Diffusion Models},
  author    = {Mehdizade, Sona},
  school    = {Freie Universit{\"a}t Berlin},
  year      = {2026}
}
```
