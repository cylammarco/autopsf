#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Example script to generate PSF"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
from astropy.io import fits
from psf import psf

# Configure the parser
parser = ArgumentParser(description="Configure seeing and pixelscale.")

parser.add_argument("filename")

parser.add_argument(
    "--convolve",
    default=True,
    help="Set to convovle the image with a gaussian before centroiding.",
)
parser.add_argument(
    "--threshold",
    default=None,
    help="the threshold of the source detection.",
)
parser.add_argument(
    "--threshold-clip-percentile",
    default=80.0,
    help="the upper limit of the data to be clipped to determine the threshold.",
)
parser.add_argument(
    "--threshold-snr",
    default=None,
    help="S/N of the source detection.",
)
parser.add_argument(
    "--seeing-keyword",
    default="SEEING",
    help="The header keyword for the seeing (arcsec)",
)
parser.add_argument(
    "--seeing",
    default=None,
    help="The seeing (arcsec).",
)
parser.add_argument(
    "--pixelscale-keyword",
    default="PIXSCALE",
    help="The header keyword for the pixel scale (arcsec per pixel).",
)
parser.add_argument(
    "--pixelscale",
    default=None,
    help="The pixel scale (arcsec per pixel).",
)
args = parser.parse_args()


extension = os.path.splitext(args.filename)[-1]
if extension in [".gz", ".bz2", ".xz"]:
    filename_no_extension = os.path.splitext(
        os.path.splitext(args.filename)[0]
    )[0]
else:
    filename_no_extension = os.path.splitext(args.filename)[0]

if extension == ".npy":
    image_data = np.load(args.filename)
    image_header = None

elif extension.lower() in [
    ".fits",
    ".fit",
    ".fts",
    ".new",
    ".gz",
    "bz2",
    "xz",
]:
    fits_hdu = fits.open(args.filename)

    # check if there is data in the first extension HDU where the image
    # should be found.
    try:
        image_data = fits_hdu[1].data
        image_header = fits_hdu[1].header

    # If not, try to PrimaryHDU. If there still isn't anything, this is not a
    # structure I will to handle. Please fix your data reduction workflow.
    except IndexError:
        image_data = fits_hdu[0].data
        image_header = fits_hdu[0].header

else:
    raise ValueError(
        "Unknown extension. It can only be npy, fits, fit, fts, new. "
        f"{extension} is given."
    )

# Assume in the unit of: arcsec
if args.seeing is None:
    try:
        seeing = float(image_header[args.seeing_keyword])

    except (KeyError, TypeError) as e:
        print(e)
        print("seeing is set to 1.5")
        seeing = 1.5

else:
    seeing = float(args.seeing)

# Assume in the unit of: arcsec per pixel
if args.pixelscale is None:
    try:
        pixel_scale = float(image_header[args.pixelscale_keyword])

    except (KeyError, TypeError) as e:
        print(e)
        print("pixel scale is set to 0.25")
        pixel_scale = 0.25

else:
    pixel_scale = float(args.pixelscale)

# Assume in the unit of: cnt per e-

# set the box size to 2 * 2 * seeing (box size has to be odd)
box_size = int(np.ceil(seeing / pixel_scale) * 4 + 1)

stars, stars_tbl = psf.get_good_stars(
    image_data,
    subtract_background=True,
    threshold=args.threshold,
    threshold_snr=args.threshold_snr,
    fwhm=seeing,
    box_size=box_size,
    npeaks=100,
    stars_tbl=None,
    edge_size=15,
    output_folder=".",
    save_stars=True,
    stars_overwrite=True,
    stars_filename=filename_no_extension + "_good_stars",
    save_stars_tbl=True,
    stars_tbl_overwrite=True,
    stars_tbl_filename=filename_no_extension + "_good_stars_tbl",
)

psf_guess, mask_list, center_list, oversampling = psf.build_psf(
    stars,
    oversampling=None,
    smoothing_kernel="quadratic",
    create_figure=True,
    save_figure=True,
    stamps_nrows=None,
    stamps_ncols=None,
    figsize=(10, 10),
    stamps_figname=filename_no_extension + "_PSF_stamps",
    psf_figname=filename_no_extension + "_PSF",
    output_folder=".",
    save_epsf_model=True,
    model_overwrite=True,
    model_filename=filename_no_extension + "_epsf_model",
    save_epsf_star=True,
    stars_overwrite=True,
    center_list_filename=filename_no_extension + "_epsf_star",
    num_iteration=100,
    n_recenter=25,
)

print(center_list)
