"""Example script for TV image denoising."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tv_denoise import (
    add_gaussian_noise,
    compute_metrics,
    load_grayscale_image,
    sweep_lambda,
    tv_denoise,
)


def main() -> None:
    image = load_grayscale_image()
    noisy = add_gaussian_noise(image, sigma=0.1, seed=0)

    lam = 0.1
    denoised = tv_denoise(noisy, lam)
    psnr, ssim = compute_metrics(image, denoised)
    print(f"lambda={lam:.3f} PSNR={psnr:.2f} SSIM={ssim:.3f}")

    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [image, noisy, denoised], ["Original", "Noisy", f"Denoised (λ={lam})"]
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()

    # Sweep over lambda values
    lambdas = np.linspace(0.01, 0.2, 5)
    results = sweep_lambda(noisy, image, lambdas.tolist())
    for lam, p, s in results:
        print(f"lambda={lam:.3f} PSNR={p:.2f} SSIM={s:.3f}")

    plt.figure()
    plt.plot(lambdas, [p for _, p, _ in results], label="PSNR")
    plt.plot(lambdas, [s for _, _, s in results], label="SSIM")
    plt.xlabel("lambda")
    plt.legend()
    plt.title("Performance vs Regularization")
    plt.show()


if __name__ == "__main__":
    main()
