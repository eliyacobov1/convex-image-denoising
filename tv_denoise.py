"""Utilities for total variation image denoising using convex optimization."""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_grayscale_image() -> np.ndarray:
    """Load the standard camera test image as a float array in [0, 1]."""
    image = img_as_float(data.camera())
    return image


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.1, seed: int | None = None) -> np.ndarray:
    """Return a noisy copy of ``image`` with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(scale=sigma, size=image.shape)
    return np.clip(noisy, 0, 1)


def tv_denoise(noisy: np.ndarray, lam: float, max_iter: int = 1000) -> np.ndarray:
    """Solve a TV-denoising problem for ``noisy`` with regularization ``lam``."""
    m, n = noisy.shape
    u = cp.Variable((m, n))
    obj = 0.5 * cp.sum_squares(u - noisy) + lam * cp.tv(u)
    constraints = [u >= 0, u <= 1]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve(verbose=False, max_iter=max_iter)
    return u.value


def compute_metrics(clean: np.ndarray, denoised: np.ndarray) -> tuple[float, float]:
    """Return PSNR and SSIM comparing ``clean`` and ``denoised``."""
    psnr = peak_signal_noise_ratio(clean, denoised, data_range=1)
    ssim = structural_similarity(clean, denoised, data_range=1)
    return psnr, ssim


def sweep_lambda(noisy: np.ndarray, clean: np.ndarray, lambdas: list[float]) -> list[tuple[float, float, float]]:
    """Denoise ``noisy`` for each lambda and return metrics.

    Returns a list of ``(lambda, psnr, ssim)`` tuples.
    """
    results = []
    for lam in lambdas:
        denoised = tv_denoise(noisy, lam)
        psnr, ssim = compute_metrics(clean, denoised)
        results.append((lam, psnr, ssim))
    return results

