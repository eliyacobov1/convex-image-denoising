# Convex Image Denoising

This project demonstrates total variation (TV) based image denoising using
convex optimization with [CVXPY](https://www.cvxpy.org/).

The example loads the ``camera`` test image from ``scikit-image``, corrupts it
with Gaussian noise and solves the optimization problem

\[ \min_I \tfrac12 \|I - I_{\text{noisy}}\|_2^2 + \lambda\,\text{TV}(I) \]

subject to \(0 \leq I \leq 1\). PSNR and SSIM are used to benchmark the
results for different regularization weights.

## Setup

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Usage

Run ``run.py`` to denoise an image and sweep across several ``lambda`` values:

```bash
python run.py
```

This displays the original, noisy and denoised images as well as a plot showing
PSNR/SSIM over ``lambda``.
