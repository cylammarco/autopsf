#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate mock images at different FWHM and pixel scales"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from photutils.datasets import make_noise_image
from scipy import signal

# set the random number seed
SEED = 123
np.random.seed(SEED)

# fixed the field of view (60 arcsec -> 1 arcmin -> 0.01666666 degree)
FOV = 180

# point-source positions
N = 50
random_x = np.random.rand(N) * FOV
random_y = np.random.rand(N) * FOV


# FWHM of the observation (arcsec)
fwhm_list = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0])

# Pixel scale (arcsec per pixel)
pixel_scale_list = np.array(
    [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 2.0, 2.5]
)


for fwhm in fwhm_list:
    if not os.path.exists(f"fwhm_{int(fwhm * 100)}"):
        os.mkdir(f"fwhm_{int(fwhm * 100)}")
    for pixel_scale in pixel_scale_list:
        if (pixel_scale > fwhm * 1.0) or (pixel_scale < fwhm / 10.0):
            continue
        # get the oberving conditions
        fwhm_in_pixel = fwhm / pixel_scale
        sigma_in_pixel = fwhm_in_pixel / 2.35
        kernel = Gaussian2DKernel(x_stddev=sigma_in_pixel)
        # Get the number of pixels to cover the given size of FOV
        n_pix = int(FOV / pixel_scale)
        frame_size = np.array([n_pix, n_pix])
        frame = np.ones(frame_size) * 50.0
        frame[
            (random_x / pixel_scale).astype("int"),
            (random_y / pixel_scale).astype("int"),
        ] = (
            np.random.random(len(random_x)) * 5e4
        )  # get the random element.
        image = signal.convolve(frame, kernel, mode="same", method="fft")
        image /= np.nanmax(image)
        image *= 5e4
        image += make_noise_image(
            image.shape,
            distribution="gaussian",
            mean=100.0,
            stddev=10.0,
            seed=SEED,
        )
        plt.figure(figsize=(8, 8))
        plt.imshow(np.log10(image), vmin=0.0, origin="lower")
        plt.tight_layout()
        plt.savefig(
            f"fwhm_{int(fwhm * 100)}/FOV_{FOV}_pixel_scale_{pixel_scale:.2f}_fwhm_{fwhm:.2f}.png"
        )
        plt.close()
        np.save(
            f"fwhm_{int(fwhm * 100)}/FOV_{FOV}_pixel_scale_{pixel_scale:.2f}_fwhm_{fwhm:.2f}.npy",
            image.astype("float16"),
        )
