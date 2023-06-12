# multi-modal-vae
UNIST AI518 Deep Generative Models 2023 Spring Group 18 Project

In this project, we implement [Multi-Modal VAE](https://arxiv.org/abs/1911.03393).

# Installation
- Install pytorch 2.0.0

- Install dependent packages
``` shell
pip install -r requirements.txt
<<<<<<< HEAD
```

# How to run

```bash
python main.py --configs <config_path_here>
```
=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89

# Configs

- `Data`,`name`: choice of `[MNIST, SVHN, MNIST-SVHN]`

# How To Run

CIFAR10:
``` shell
python main.py --config configs/cifar10.yaml
<<<<<<< HEAD
```

Generate samples from trained model:
``` shell
python generate.py --config configs/mnist.yaml --model_path {path/to/model.pt} --num_samples 1000
=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89
```