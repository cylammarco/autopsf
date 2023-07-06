#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from photutils.aperture import ApertureStats, CircularAperture
from photutils.background import (
    Background2D,
    BiweightLocationBackground,
    MeanBackground,
    MedianBackground,
    MMMBackground,
    ModeEstimatorBackground,
    SExtractorBackground,
)
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint
from psfr.psfr import stack_psf
from scipy import ndimage, signal
from scipy.optimize import curve_fit

__all__ = ["get_background", "get_good_stars", "build_psf"]


def _gaus(X2, C, sigma):
    # X2 is (X - X_mean)**2.0
    return C * np.exp(-X2 / (2 * sigma**2))


def get_background(
    image_data,
    image_header,
    sigma=3.0,
    background_estimator="MMM",
    box_size=51,
    filter_size=5,
    output_folder=".",
    background_filename="background",
    background_subtracted_filename="background_subtracted",
    save_figure=True,
    save_fits=True,
    fig_size=(10, 10),
    **bkg_kwargs,
):
    sigma_clip = SigmaClip(sigma=sigma)

    if background_estimator == "MMM":
        bkg_estimator = MMMBackground(sigma_clip=sigma_clip, **bkg_kwargs)
    elif background_estimator == "mean":
        bkg_estimator = MeanBackground(sigma_clip=sigma_clip, **bkg_kwargs)
    elif background_estimator == "median":
        bkg_estimator = MedianBackground(sigma_clip=sigma_clip, **bkg_kwargs)
    elif background_estimator == "Mode":
        bkg_estimator = ModeEstimatorBackground(
            sigma_clip=sigma_clip, **bkg_kwargs
        )
    elif background_estimator == "Biweight":
        bkg_estimator = BiweightLocationBackground(
            sigma_clip=sigma_clip, **bkg_kwargs
        )
    elif background_estimator == "SE":
        bkg_estimator = SExtractorBackground(
            sigma_clip=sigma_clip, **bkg_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported background estimator {background_estimator}. "
            "Please choose from MMM, mean, median and SE"
        )

    threshold_2D = detect_threshold(
        image_data, nsigma=3.0, sigma_clip=sigma_clip
    )
    segment_img = detect_sources(image_data, threshold_2D, npixels=10)
    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    bkg = Background2D(
        image_data,
        (box_size, box_size),
        filter_size=(
            filter_size,
            filter_size,
        ),
        mask=mask,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )

    bkg_norm = simple_norm(bkg.background, "log", percent=95.0)
    img_norm = simple_norm(image_data, "log", percent=99.0)

    # get the background subtracted image
    image_data -= bkg.background

    if save_figure:
        # save the background
        plt.figure(1, fig_size)
        plt.clf()
        plt.imshow(
            bkg.background, origin="lower", aspect="auto", norm=bkg_norm
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, background_filename))

        # save the background subtracted image
        plt.figure(2, fig_size)
        plt.clf()
        plt.imshow(image_data, origin="lower", aspect="auto", norm=img_norm)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, background_subtracted_filename)
        )

    if save_fits:
        background_subtracted_fits = fits.PrimaryHDU(image_data)
        background_subtracted_fits.header = image_header
        background_subtracted_fits.writeto(
            os.path.join(
                output_folder,
                background_subtracted_filename + ".fits",
            ),
            overwrite=True,
        )

    return image_data, bkg


def get_good_stars(
    image_data,
    threshold=None,
    threshold_snr=3.0,
    threshold_sigma_clip=3.0,
    box_size=35,
    fwhm_pix=None,
    minsep_fwhm=2.5,
    convolve=True,
    save_convolved_figure=True,
    fig_size=(10, 10),
    convolved_figure_filename="convolved_image",
    sigma=None,
    stars_tbl=None,
    edge_size=25,
    output_folder=".",
    save_dao_tbl=True,
    dao_tbl_filename="dao_tbl",
    save_stars=True,
    stars_overwrite=True,
    stars_filename="good_stars",
    save_stars_tbl=True,
    stars_tbl_overwrite=True,
    stars_tbl_filename="good_stars_tbl",
    save_good_stars_figure=True,
    **kwargs,
):
    """
    Get the centroids of the bright sources to prepare to compute the FWHM.
    The data should be background subtracted, and cosmic ray cleaned.

    Parameters
    ----------
    image_data: array_like
        The 2D array of the image.
    threshold: float or array-like (Default: None)
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D threshold must have the same shape as
        data. See photutils.detection.detect_threshold for one way to
        create a threshold image.
    box_size: scalar or tuple, optional (Default: 15)
        The size of the local region to search for peaks at every point in
        data. If box_size is a scalar, then the region shape will be
        (box_size, box_size). Either box_size or footprint must be defined.
        If they are both defined, then footprint overrides box_size.
    fwhm_pix: float (Default: None)
        The full with at half maximum in unit of pixels. If not provided,
        it takes the size 1/8 of the box_size
    minsep_fwhm: float (Default: 3.0)
        The minimum separation for detected objects in units of fwhm.
    convolve: bool (Default: True)
        Set to True to first convovle the image with a Gaussian kernel of
        size sigma (next argument). This allows better centroiding of
        defocused images. This is ONLY USED for centroiding, the building of
        the PSF will be using the input data.
    sigma: float (Default: None)
        Defining the size of the Gaussian kernel, if not provided, it takes
        the value of fwhm/2.355.
    stars_tbl: Table, list of Table, optional (Default: None)
        A catalog or list of catalogs of sources to be extracted from the
        input data. To link stars in multiple images as a single source,
        you must use a single source catalog where the positions defined in
        sky coordinates.
        If a list of catalogs is input (or a single catalog with a single
        NDData object), they are assumed to correspond to the list of NDData
        objects input in data (i.e., a separate source catalog for each 2D
        image). For this case, the center of each source can be defined either
        in pixel coordinates (in x and y columns) or sky coordinates (in a
        skycoord column containing a SkyCoord object). If both are specified,
        then the pixel coordinates will be used.
        If a single source catalog is input with multiple NDData objects, then
        these sources will be extracted from every 2D image in the input data.
        In this case, the sky coordinates for each source must be specified as
        a SkyCoord object contained in a column called skycoord. Each NDData
        object in the input data must also have a valid wcs attribute. Pixel
        coordinates (in x and y columns) will be ignored.
        Optionally, each catalog may also contain an id column representing the
        ID/name of stars. If this column is not present then the extracted
        stars will be given an id number corresponding the the table row number
        (starting at 1). Any other columns present in the input catalogs will
        be ignored.
    edge_size: int (Default: 50)
        The number of pixels from the detector edges to be removed.
    output_folder: str (Default: ".")
        The base folder where the files will be saved. Defaulted to the current
        directory.
    save_dao_tbl:

    dao_tbl_filename:

    save_stars: bool (Default: True)
        Set to True to save the extracted stars, an EPSFStars instance
        (output of photutils.extract_stars())
    stars_overwrite: bool (Default: True)
        Set to True to overwrite existing star file.
    stars_filename: str (Default: "good_stars")
        Filename of the EPSFStars instance.
    save_stars_tbl: bool (Default: True)
        Set to True to save the star_tbl.
    stars_tbl_overwrite: bool (Default: True)
        Set to True to overwrite existing star_tbl file.
    stars_tbl_filename: str (Default: "good_stars_tbl")
        Filename of the star_tbl.
    **kwargs:
        keyword arguments for DAOStarFinder.

    Return
    ------
    stars: EPSFStars instance
        A photutils.psf.EPSFStars instance containing the extracted stars.
    stars_tbl: Table, list of Table
        A table containing the x and y pixel location of the peaks and their
        values. If centroid_func is input, then the table will also contain the
        centroid position. If no peaks are found then None is returned.

    """

    if threshold is None:
        sigma_clip = SigmaClip(sigma=threshold_sigma_clip)
        threshold_2D = detect_threshold(
            image_data, nsigma=3.0, sigma_clip=sigma_clip
        )
        if sigma is None:
            if fwhm_pix is None:
                sigma = (box_size / 8) / 2.355
            else:
                sigma = fwhm_pix / 2.355
        segment_img = detect_sources(
            image_data, threshold_2D, npixels=int(3.1416 * sigma**2.0)
        )
        footprint = circular_footprint(radius=15)
        mask = segment_img.make_source_mask(footprint=footprint)
        mean, _, std = sigma_clipped_stats(image_data, sigma=3.0, mask=mask)
        threshold = mean + std * threshold_snr
    else:
        threshold = float(threshold)

    print(f"threshold is set at {threshold}.")

    # Convolve with 2D gaussian to improve centroiding
    # First a 1-D  Gaussian
    if convolve:
        t = np.linspace(-box_size, box_size, box_size * 5)
        bump = np.exp(-0.5 * (t / sigma) ** 2)
        bump /= np.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

        data_convolved = signal.fftconvolve(image_data, kernel, mode="same")

        if save_convolved_figure:
            convolved_img_norm = simple_norm(
                data_convolved[edge_size:-edge_size, edge_size:-edge_size],
                "log",
                percent=99.0,
            )

            plt.figure(1, fig_size)
            plt.clf()
            plt.imshow(
                data_convolved,
                origin="lower",
                aspect="auto",
                norm=convolved_img_norm,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, convolved_figure_filename))

    else:
        data_convolved = image_data

    if stars_tbl is None:
        # First iteration to get the FWHM if fwhm_pix was None
        if fwhm_pix is None:
            _fwhm_pix = box_size / 8.0

            # detect peaks and remove sources near the edge
            daofind = DAOStarFinder(
                fwhm=_fwhm_pix, threshold=threshold, **kwargs
            )
            peaks_tbl = daofind(
                data_convolved[edge_size:-edge_size, edge_size:-edge_size]
            )
            peaks_sort_mask = np.argsort(-peaks_tbl["flux"])
            peaks_tbl = peaks_tbl[peaks_sort_mask]
            _x = peaks_tbl["xcentroid"] + edge_size
            _y = peaks_tbl["ycentroid"] + edge_size

            aper = CircularAperture(zip(_x, _y), _fwhm_pix * 2)
            aperstats = ApertureStats(image_data, aper)
            fwhm_pix = np.median(aperstats.fwhm.value)
            box_size = int((fwhm_pix * 8 // 2) * 2 + 1)

            if convolve:
                sigma = fwhm_pix / 2.355
                t = np.linspace(-box_size, box_size, box_size * 5)
                bump = np.exp(-0.5 * (t / sigma) ** 2)
                bump /= np.trapz(bump)  # normalize the integral to 1

                # make a 2-D kernel out of it
                kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

                data_convolved = signal.fftconvolve(
                    image_data, kernel, mode="same"
                )

        else:
            fwhm_pix = float(fwhm_pix)

        # Get the stars here
        daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=threshold, **kwargs)
        peaks_tbl = daofind(
            data_convolved[edge_size:-edge_size, edge_size:-edge_size]
        )
        peaks_sort_mask = np.argsort(-peaks_tbl["flux"])
        peaks_tbl = peaks_tbl[peaks_sort_mask]
        _x = peaks_tbl["xcentroid"]
        _y = peaks_tbl["ycentroid"]

        coord = np.column_stack((_x, _y))
        distances = np.array(
            [np.linalg.norm(coord - p, axis=1) for p in coord]
        )
        mask = distances < fwhm_pix * minsep_fwhm
        # Ignore distnace to itself
        for i, _ in enumerate(_x):
            mask[i][i] = False

        mask = np.sum(mask, axis=0).astype("bool")
        peaks_tbl = peaks_tbl[~mask]

        x = peaks_tbl["xcentroid"]
        y = peaks_tbl["ycentroid"]

        if save_dao_tbl:
            dao_tbl_output_path = os.path.join(
                output_folder, dao_tbl_filename + ".npy"
            )
            np.save(dao_tbl_output_path, peaks_tbl)

        stars_tbl = Table()
        stars_tbl["x"] = x + edge_size
        stars_tbl["y"] = y + edge_size

        if save_stars_tbl:
            stars_tbl_output_path = os.path.join(
                output_folder, stars_tbl_filename + ".npy"
            )
            if os.path.exists(stars_tbl_output_path) and (
                not stars_tbl_overwrite
            ):
                print(
                    stars_tbl_output_path + " already exists. Use a "
                    "different name or set overwrite to True. EPSFModel is "
                    "not saved to disk."
                )
            else:
                np.save(stars_tbl_output_path, stars_tbl)

    nddata = NDData(data=image_data)

    stars = extract_stars(nddata, catalogs=stars_tbl, size=box_size)

    if save_stars:
        stars_output_path = os.path.join(
            output_folder, stars_filename + ".pbl"
        )
        if os.path.exists(stars_output_path) and (not stars_overwrite):
            print(
                stars_output_path + " already exists. Use a different "
                "name or set overwrite to True. EPSFStar is not saved to "
                "disk."
            )
        else:
            with open(stars_output_path, "wb+") as f:
                pickle.dump(stars, f)

    if save_good_stars_figure:
        img_norm = simple_norm(
            image_data[edge_size:-edge_size, edge_size:-edge_size],
            "log",
            percent=99.0,
        )
        img_width, img_height = np.shape(image_data)

        plt.figure(1, fig_size)
        plt.clf()
        plt.imshow(image_data, origin="lower", aspect="auto", norm=img_norm)
        plt.scatter(
            stars_tbl["x"],
            stars_tbl["y"],
            s=box_size / 2,
            facecolors="none",
            edgecolors="r",
        )
        plt.plot(
            [
                edge_size,
                edge_size,
                img_width - edge_size,
                img_width - edge_size,
                edge_size,
            ],
            [
                edge_size,
                img_height - edge_size,
                img_height - edge_size,
                edge_size,
                edge_size,
            ],
            c="black",
            ls=":",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, stars_filename))

    return stars, stars_tbl, threshold, fwhm_pix


def build_psf(
    stars,
    oversampling=None,
    return_oversampled=False,
    create_figure=True,
    save_figure=True,
    stamps_nrows=None,
    stamps_ncols=None,
    figsize=(10, 10),
    stamps_figname="PSF_stamps",
    psf_figname="PSF",
    output_folder=".",
    save_psf_model=True,
    model_overwrite=True,
    model_filename="psf_model",
    save_center_list=True,
    center_list_overwrite=True,
    center_list_filename="psf_center_list",
    **kwargs,
):
    """
    PSF is built using the 'stars' provided, but the stamps_nrows and
    stamps_ncols are only used for controlling the display

    Parameters
    ----------
    stars: EPSFStars instance
        A photutils.psf.EPSFStars instance containing the extracted stars.
    oversampling: int or tuple of two int, optional(Default: None)
        The oversampling factor(s) of the ePSF relative to the input stars
        along the x and y axes. The oversampling can either be a single float
        or a tuple of two floats of the form (x_oversamp, y_oversamp). If
        oversampling is a scalar then the oversampling will be the same for
        both the x and y axes.
    create_figure: bool (Default: True)
        Create and display the cutouts of the regions used for building the
        PSF.
    save_figure: bool (Default: True)
        Save the figure, only possible if it is created.
    stamps_nrows: (Default: None)
        Number of rows to display. This does NOT affect the number of stars
        used for building the PSF.
    stamps_ncols: (Default: None)
        Number of columns to display. This does NOT affect the number of stars
        used for building the PSF.
    figsize: (Default: (10, 10))
        Figure size.
    stamps_figname: str (Default: "PSF_stamps")
        Name of the output figures.
    psf_figname: str (Default: "PSF")
        Name of the output figures.
    output_folder: str (Default: ".")
        The base folder where the files will be saved. Defaulted to the current
        directory.
    save_psf_model: bool (Default: True)
        Save the effective PSF as npy.
    model_overwrite: bool (Default: True)
        Set to True to overwrite.
    model_filename: str (Default: "psf_model")
        Filename of the effective PSF model.
    save_center_list: bool (Default: True)
        Save the centroid position of the points used to build the PSF. The
        list can be slightly different to the position of the input star_tbl
        positions because psfr performs recentering during the building of
        the PSF.
    center_list_overwrite: bool (Default: True)
        Set to True to overwrite.
    center_list_filename: str (Default: "psf_center_list")
        Filename of the center_list.
    **kwargs
        Extra arguments for psfr.stack_psf().

    Return
    ------
    psf_guess: numpy.ndarray
        The constructed effective PSF array in the oversampled resolution.
    mask_list:
        List of masks for each individual star's pixel to be included in the
        fit or not. 0 means excluded, 1 means included. This list is updated
        with all the criteria applied on the fitting and might deviate from
        the input mask_list.
    center_list:
        List of astrometric centers relative to the center pixel of the
        individual stars.
    oversampling:
        Return the oversampling factor used. It can be different from the
        input because if the input is too large, the factor will be reduced
        automatically.

    """

    if oversampling is None:
        oversampling = int(np.sqrt(len(stars))) - 1
        oversampling = oversampling - oversampling % 2
        if oversampling < 2:
            oversampling = 2

    oversampling = int(oversampling)

    # Build the PSFs
    postage_stamps = [i.data for i in stars]
    psf_guess, center_list, mask_list = stack_psf(
        postage_stamps, oversampling=oversampling, **kwargs
    )

    if not return_oversampled:
        postage_stamp_size = len(postage_stamps[0][0])
        psf_guess_size = len(psf_guess[0])
        psf_guess = ndimage.zoom(
            psf_guess, postage_stamp_size / psf_guess_size, grid_mode=True
        )

    if create_figure:
        n_star = len(postage_stamps)

        # Get the nearest square number to fill if the number of rows and/or
        # columns is/are not provided.
        if (stamps_nrows is None) and (stamps_ncols is None):
            min_sq_number = int(np.ceil(np.sqrt(n_star)))
            stamps_nrows = min_sq_number
            stamps_ncols = min_sq_number

        elif stamps_nrows is None:
            stamps_nrows = int(np.ceil(n_star / stamps_ncols))

        elif stamps_ncols is None:
            stamps_ncols = int(np.ceil(n_star / stamps_nrows))

        else:
            pass

        # Set up the figure
        _, axes = plt.subplots(
            nrows=stamps_nrows,
            ncols=stamps_ncols,
            figsize=figsize,
            squeeze=True,
        )

        if stamps_nrows > 1:
            axes = axes.ravel()

        else:
            axes = [axes]

        for i in range(int(min_sq_number**2)):
            if i < n_star:
                norm = simple_norm(postage_stamps[i], "log", percent=95.0)
                axes[i].imshow(
                    postage_stamps[i],
                    norm=norm,
                    origin="lower",
                    cmap="viridis",
                )
                axes[i].set_xticklabels([""])
                axes[i].set_yticklabels([""])

            else:
                axes[i].axis("off")

        if save_figure:
            plt.savefig(os.path.join(output_folder, stamps_figname))

        epsf_norm = simple_norm(psf_guess, "log", percent=95.0)

        plt.figure()
        plt.imshow(psf_guess, norm=epsf_norm, origin="lower")
        plt.colorbar()

        if save_figure:
            plt.savefig(os.path.join(output_folder, psf_figname))

    if save_psf_model:
        model_output_path = os.path.join(output_folder, model_filename)

        if os.path.exists(model_output_path) and (not model_overwrite):
            print(
                model_output_path
                + " already exists. Use a different name or set overwrite "
                "to True. PSF model is not saved to disk."
            )

        else:
            np.save(model_output_path, psf_guess)
            fits.PrimaryHDU(psf_guess).writeto(
                model_output_path + ".fits", overwrite=True
            )

    if save_center_list:
        center_list_output_path = os.path.join(
            output_folder, center_list_filename
        )

        if os.path.exists(center_list_output_path) and (
            not center_list_overwrite
        ):
            print(
                center_list_output_path
                + " already exists. Use a different name or set overwrite "
                "to True. center_list is not saved to disk."
            )

        else:
            np.save(center_list_output_path, center_list)

    return psf_guess, mask_list, center_list, oversampling
