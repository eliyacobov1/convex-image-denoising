import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tv_denoise import load_grayscale_image, add_gaussian_noise, tv_denoise, compute_metrics


def test_tv_improves_psnr():
    image = load_grayscale_image()
    image = image[::4, ::4]
    noisy = add_gaussian_noise(image, sigma=0.1, seed=0)

    psnr_noisy, _ = compute_metrics(image, noisy)
    denoised = tv_denoise(noisy, lam=0.1)
    psnr_denoised, _ = compute_metrics(image, denoised)

    assert psnr_denoised > psnr_noisy
    assert denoised.shape == image.shape
