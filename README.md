# Laplace-Bridged Randomized Smoothing for Fast Certified Robustness

## Overview

Randomized Smoothing (RS) provides formal certified robustness guarantees under $\ell_p$ perturbations for arbitrary base classifiers.  
However, existing RS approaches suffer from two major practical limitations:

1. **Dependence on noise-augmented training.**  
   In practice, RS often requires training the base classifier with injected noise to obtain nontrivial certificates. This increases training cost, may reduce clean accuracy, and weakens RS as a genuinely post-hoc defense.

2. **High certification cost.**  
   RS certification typically requires tens of thousands of noisy forward passes per input, which severely limits deployment, especially on resource-constrained edge devices.

To address both issues, we propose **Laplace-Bridged Smoothing (LBS)**, a fast and post-hoc certified robustness framework that significantly reduces certification cost while eliminating reliance on noise-augmented training.

## Base Classifier Training (RS)

The training of the base classifier in this repository **directly builds upon the RS implementation** from:

> **Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019**  
> https://github.com/locuslab/smoothing

The directory `smoothing-based/` is adapted from this codebase.

### Training Configuration

Since **LBS is designed to address the dependence of RS on noise-augmented training**, we train the base classifier **without noise injection**, i.e.,

```
python smoothing-based/train.py cifar10 cifar_resnet110  models/cifar10/resnet110/noise_0.00 --batch 400 --noise 0.00 --gpu [num]

python smoothing-based/train.py imagenet imagenet_vit  models/imagenet/vit/noise_0.00 --batch 400 --noise 0.0

```

## LBS Inference and Certification

LBS inference and certification are performed **post hoc**, independently of the base classifier training procedure.
After a base classifier is trained without noise augmentation, LBS can be directly applied to compute certified robustness under different $\ell_p$ threat models.

The main entry point for LBS inference and certification is:

```
python lbs-inference/certification_l2.py

python lbs-inference/certification_l1.py

python lbs-inference/certification_l_infinity.py
```
## Plotting and Visualization

To visualize certified robustness results produced by LBS, using:

```
python lbs-inference/plot.py
```


