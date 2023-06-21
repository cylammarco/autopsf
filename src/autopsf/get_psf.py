#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Example script to generate PSF"""

import os
import sys

import numpy as np
import psf
import yaml
from astropy.io import fits
from astropy.stats import SigmaClip
from astroscrappy import detect_cosmics
from photutils.background import Background2D, MMMBackground

# This returns where the Python instance started, not where this script is
HERE = os.getcwd()

# Get config file
filename = sys.argv[1]
param_filename = sys.argv[2]
params_path = os.path.join(HERE, param_filename)
input_file_path = os.path.dirname(os.path.abspath(filename))

if not os.path.isabs(params_path):
    params_path = os.path.abspath(params_path)

print("Reading parameters from " + params_path + ".")

with open(params_path, "r") as stream:
    params = yaml.safe_load(stream)


extension = os.path.splitext(filename)[-1]

# remove the extensions in the filename
if extension in [".gz", ".bz", ".bz2", ".xz", ".fz"]:
    filename_no_extension = os.path.splitext(os.path.splitext(filename)[0])[0]
else:
    filename_no_extension = os.path.splitext(filename)[0]

# remove folders in the path
filename_no_extension = filename_no_extension.split(os.sep)[-1]

if extension == ".npy":
    image_data = np.load(filename)
    image_header = None

elif extension.lower() in [
    ".fits",
    ".fit",
    ".fts",
    ".new",
    ".gz",
    ".bz",
    ".bz2",
    ".fz",
    ".xz",
]:
    fits_hdu = fits.open(filename)

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


if params["get_good_stars_output_folder"] is None:
    get_good_stars_output_folder = input_file_path
else:
    get_good_stars_output_folder = params["get_good_stars_output_folder"]

if params["build_psf_output_folder"] is None:
    build_psf_output_folder = input_file_path
else:
    build_psf_output_folder = params["build_psf_output_folder"]

lowercase_header = [x.lower() for x in image_header]

# Assume in the unit of: arcsec
if params["seeing"] is None:
    if isinstance(params["seeing_keyword"], str):
        seeing = float(image_header[params["seeing_keyword"]])
        print(f"seeing is set to {seeing}")
    else:
        if np.in1d(lowercase_header, params["seeing_keyword"]).any():
            # Get the exposure time for the light frames
            seeing_keyword_idx = int(
                np.where(np.in1d(params["seeing_keyword"], lowercase_header))[
                    0
                ][0]
            )
            seeing_keyword = params["seeing_keyword"][seeing_keyword_idx]
            seeing = float(image_header[seeing_keyword])
            print(f"seeing is set to {seeing}")
        else:
            seeing = 1.5
            print("seeing is set to 1.5")
else:
    seeing = float(params["seeing"])
    print(f"seeing is set to {seeing}")

# Assume in the unit of: arcsec per pixel
if params["pixelscale"] is None:
    if isinstance(params["pixelscale_keyword"], str):
        pixel_scale = float(image_header[params["pixelscale_keyword"]])
        print(f"pixel_scale is set to {pixel_scale}")
    else:
        if np.in1d(lowercase_header, params["pixelscale_keyword"]).any():
            # Get the exposure time for the light frames
            pixelscale_keyword_idx = int(
                np.where(
                    np.in1d(params["pixelscale_keyword"], lowercase_header)
                )[0][0]
            )
            pixelscale_keyword = params["pixelscale_keyword"][
                pixelscale_keyword_idx
            ]
            pixel_scale = float(image_header[pixelscale_keyword])
            print(f"pixel_scale is set to {pixel_scale}")
        else:
            pixel_scale = 0.25
            print("pixel scale is set to 0.25")
else:
    pixel_scale = float(params["pixelscale"])
    print(f"pixel_scale is set to {pixel_scale}")

# Assume in the unit of: cnt per e-
if params["remove_cr"]:
    image_data = detect_cosmics(image_data, **params["kwargs_remove_cr"])[1]

if params["subtract_background"]:
    sigma_clip = SigmaClip(sigma=params["sigma_clip_sigma"])
    bkg_estimator = MMMBackground()
    bkg = Background2D(
        image_data,
        (params["background_box_size"], params["background_box_size"]),
        filter_size=(
            params["background_filter_size"],
            params["background_filter_size"],
        ),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    image_data -= bkg.background

# set the box size to 2 * 2 * seeing (box size has to be odd)
box_size = int(np.ceil(seeing / pixel_scale) * 4 + 1)

if params["stars_filename"] is None:
    stars_filename = filename_no_extension + "_good_stars"
else:
    stars_filename = params["stars_filename"]

if params["stars_tbl_filename"] is None:
    stars_tbl_filename = filename_no_extension + "_good_stars_tbl"
else:
    stars_tbl_filename = params["stars_tbl_filename"]


stars, stars_tbl = psf.get_good_stars(
    image_data,
    threshold=params["threshold"],
    threshold_snr=params["threshold_snr"],
    fwhm=seeing,
    box_size=box_size,
    npeaks=params["npeaks"],
    stars_tbl=params["stars_tbl"],
    edge_size=params["edge_size"],
    output_folder=get_good_stars_output_folder,
    save_stars=params["save_stars"],
    stars_overwrite=params["stars_overwrite"],
    stars_filename=stars_filename,
    save_stars_tbl=params["save_stars_tbl"],
    stars_tbl_overwrite=params["stars_tbl_overwrite"],
    stars_tbl_filename=stars_tbl_filename,
    **params["get_good_stars_kwargs"],
)


if params["stamps_figname"] is None:
    stamps_figname = filename_no_extension + "_psf_stamps"
else:
    stamps_figname = params["stamps_figname"]

if params["psf_figname"] is None:
    psf_figname = filename_no_extension + "_psf"
else:
    psf_figname = params["psf_figname"]

if params["model_filename"] is None:
    model_filename = filename_no_extension + "_psf_model"
else:
    model_filename = params["model_filename"]

if params["center_list_filename"] is None:
    center_list_filename = filename_no_extension + "_center_list"
else:
    center_list_filename = params["center_list_filename"]


psf_guess, mask_list, center_list, oversampling = psf.build_psf(
    stars,
    oversampling=params["oversampling"],
    return_oversampled=params["return_oversampled"],
    smoothing_kernel=params["smoothing_kernel"],
    create_figure=params["create_figure"],
    save_figure=params["save_figure"],
    stamps_nrows=params["stamps_nrows"],
    stamps_ncols=params["stamps_ncols"],
    figsize=params["figsize"],
    stamps_figname=stamps_figname,
    psf_figname=psf_figname,
    output_folder=build_psf_output_folder,
    save_psf_model=params["save_psf_model"],
    model_overwrite=params["model_overwrite"],
    model_filename=model_filename,
    save_center_list=params["save_center_list"],
    center_list_overwrite=params["center_list_overwrite"],
    center_list_filename=center_list_filename,
    **params["build_psf_kwargs"],
)
