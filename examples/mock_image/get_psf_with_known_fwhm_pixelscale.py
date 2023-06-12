#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Example script to generate PSF"""

import glob
import os

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MMMBackground
from psf import psf

# def main():

folder_list = sorted(glob.glob("fwhm*"))

for folder in folder_list:
    print(folder)
    filelist = glob.glob(f"{folder}{os.sep}*[0-9].npy")

    for filepath in filelist:
        filename = filepath.split(os.sep)[-1]
        print(filename)

        image_data = np.load(filepath).astype("float64")

        filepath_body = os.path.splitext(filename)[0]

        # arcsec
        fwhm = float(os.path.splitext(filename.split("_")[6])[0])

        # assuming seeing is matched to fwhm
        seeing = fwhm * 1.0
        print(f"Working with seeing: {seeing}")

        # arcsec per pixel
        pixel_scale = float(filename.split("_")[4])
        print(f"Working with pixel-scale: {pixel_scale}")

        # set the box size to 4 * seeing + 1 (box size has to be odd)
        box_size = int(np.ceil(seeing / pixel_scale) * 4 // 2 * 2 + 1)
        filter_size = int(seeing / pixel_scale * 2) // 2 * 2 + 1

        sigma_clip = SigmaClip(sigma=2.0)
        bkg_estimator = MMMBackground()
        bkg = Background2D(
            image_data,
            (filter_size, filter_size),
            filter_size=(filter_size, filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        image_data -= bkg.background

        stars, stars_tbl = psf.get_good_stars(
            image_data,
            threshold=bkg.background_rms_median * 8.0,
            fwhm=fwhm / pixel_scale,
            box_size=box_size,
            npeaks=100,
            stars_tbl=None,
            edge_size=box_size,
            output_folder=".",
            save_stars=True,
            stars_overwrite=True,
            stars_filename=f"{folder}{os.sep}{filepath_body}_good_stars",
            save_stars_tbl=True,
            stars_tbl_overwrite=True,
            stars_tbl_filename=f"{folder}{os.sep}{filepath_body}_good_stars_tbl",
        )

        psf.build_psf(
            stars,
            oversampling=None,
            smoothing_kernel="quadratic",
            create_figure=True,
            save_figure=True,
            stamps_nrows=None,
            stamps_ncols=None,
            figsize=(10, 10),
            output_folder=".",
            save_epsf_model=True,
            model_overwrite=True,
            model_filename=f"{folder}{os.sep}{filepath_body}_epsf_model",
            save_epsf_star=True,
            stars_overwrite=True,
            stars_filename=f"{folder}{os.sep}{filepath_body}_epsf_star",
            norm_radius=box_size / 2,
            sigma_clip=sigma_clip,
            stamps_figname=f"{folder}{os.sep}{filepath_body}_psf_stamps.png",
            psf_figname=f"{folder}{os.sep}{filepath_body}_epsf.png",
            num_iteration=50,
        )


if __name__ == "__main__":
    main()
