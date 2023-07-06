#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Example script to generate PSF"""

import os
import sys

import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.visualization import simple_norm
from autopsf import psf
from ccdproc import cosmicray_lacosmic

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


if params["output_folder"] is None:
    output_folder = input_file_path
else:
    output_folder = params["output_folder"]

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
            seeing = None
            print('seeing is set to 1.5"')
else:
    seeing = float(params["seeing"])
    print(f'seeing is set to {seeing}"')

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


n_fwhm = params["n_fwhm"]

if params["fwhm_pix"] is None:
    if seeing is not None:
        fwhm_pix = seeing / pixel_scale

    else:
        fwhm_pix = None

else:
    fwhm_pix = float(params["fwhm_pix"])

# Assume in the unit of: cnt per e-
if params["remove_cr"]:
    image_data = cosmicray_lacosmic(
        image_data, gain_apply=False, **params["kwargs_remove_cr"]
    )[0]

if params["subtract_background"]:
    # get the background file name
    if params["background_filename"] is None:
        background_filename = filename_no_extension + "_background"
    else:
        background_filename = params["background_filename"]

    # get the background file name
    if params["background_subtracted_filename"] is None:
        background_subtracted_filename = (
            filename_no_extension + "_background_subtracted"
        )
    else:
        background_subtracted_filename = params[
            "background_subtracted_filename"
        ]

    image_data, bkg = psf.get_background(
        image_data,
        image_header,
        sigma=params["sigma_clip"],
        background_estimator=params["background_estimator"],
        box_size=params["background_box_size"],
        filter_size=params["background_filter_size"],
        output_folder=output_folder,
        background_filename=background_filename,
        background_subtracted_filename=background_subtracted_filename,
        save_figure=params["save_background_figure"],
        save_fits=params["save_background_fits"],
        fig_size=params["background_figsize"],
        **params["bkg_kwargs"],
    )

    if params["set_saturation_to_zero"]:
        saturation_mask = image_data > params["kwargs_remove_cr"]["satlevel"]
        image_data[saturation_mask] = 0.0

# set the box size to n_fwhm * seeing (box size has to be odd)
if params["box_size"] is None:
    if not isinstance(n_fwhm, (int, float)):
        n_fwhm = 8

    if seeing is None:
        box_size = int(np.round(3.0 / pixel_scale) * n_fwhm + 1)

    else:
        box_size = int(np.round(seeing / pixel_scale) * n_fwhm + 1)

else:
    box_size = int(params["box_size"])
    if box_size % 2 == 0:
        box_size += 1

if params["dao_tbl_filename"] is None:
    dao_tbl_filename = filename_no_extension + "_dao_tbl"
else:
    dao_tbl_filename = params["dao_tbl_filename"]

if params["stars_filename"] is None:
    stars_filename = filename_no_extension + "_good_stars"
else:
    stars_filename = params["stars_filename"]

if params["stars_tbl_filename"] is None:
    stars_tbl_filename = filename_no_extension + "_good_stars_tbl"
else:
    stars_tbl_filename = params["stars_tbl_filename"]

if params["convolved_figure_filename"] is None:
    convolved_figure_filename = filename_no_extension + "_convolved"
else:
    convolved_figure_filename = params["convolved_figure_filename"]

stars, stars_tbl, threshold, fwhm_pix = psf.get_good_stars(
    image_data,
    threshold=params["threshold"],
    threshold_snr=params["threshold_snr"],
    threshold_sigma_clip=params["threshold_sigma_clip"],
    box_size=box_size,
    fwhm_pix=fwhm_pix,
    minsep_fwhm=params["minsep_fwhm"],
    convolve=params["convolve"],
    save_convolved_figure=params["save_convolved_figure"],
    fig_size=params["fig_size"],
    convolved_figure_filename=convolved_figure_filename,
    sigma=params["sigma"],
    stars_tbl=params["stars_tbl"],
    edge_size=params["edge_size"],
    output_folder=output_folder,
    save_dao_tbl=params["save_dao_tbl"],
    dao_tbl_filename=dao_tbl_filename,
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
    create_figure=params["create_figure"],
    save_figure=params["save_figure"],
    stamps_nrows=params["stamps_nrows"],
    stamps_ncols=params["stamps_ncols"],
    figsize=params["figsize"],
    stamps_figname=stamps_figname,
    psf_figname=psf_figname,
    output_folder=output_folder,
    save_psf_model=params["save_psf_model"],
    model_overwrite=params["model_overwrite"],
    model_filename=model_filename,
    save_center_list=params["save_center_list"],
    center_list_overwrite=params["center_list_overwrite"],
    center_list_filename=center_list_filename,
    **params["build_psf_kwargs"],
)
