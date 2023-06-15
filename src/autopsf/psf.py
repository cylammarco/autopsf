#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars
from psfr.psfr import stack_psf
from scipy import ndimage, signal

__all__ = ["get_good_stars", "build_psf"]


def get_good_stars(
    data,
    threshold=None,
    threshold_snr=None,
    threshold_clip_percentile=80.0,
    box_size=25,
    fwhm=None,
    convolve=True,
    sigma=None,
    npeaks=100,
    stars_tbl=None,
    edge_size=15,
    output_folder=".",
    save_stars=True,
    stars_overwrite=True,
    stars_filename="good_stars",
    save_stars_tbl=True,
    stars_tbl_overwrite=True,
    stars_tbl_filename="good_stars_tbl",
    **kwargs,
):
    """
    Get the centroids of the bright sources to prepare to compute the FWHM.
    The data should be background subtracted, and cosmic ray cleaned.

    Parameters
    ----------
    data: array_like
        The 2D array of the image.
    threshold: float or array-like (Default: None)
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D threshold must have the same shape as
        data. See photutils.detection.detect_threshold for one way to
        create a threshold image. This overrides the next two arguments
        threshold_snr and threshold_clip_percentile.
    threshold_snr: float (Default: None)
        If a threshold is not provided, it will be estimated from the image
        by measuring the standard deviation of the image excluding the values
        lower than the threshold_clip_percentile percentile. This SNR is the
        multiplier to this standard deviation to define the threshold.
    threshold_clip_percentile: float (Default: 80.0)
        The percentile of the image value that will be clipped for detemining
        the standard deviation of the image background.
    box_size: scalar or tuple, optional (Default: 15)
        The size of the local region to search for peaks at every point in
        data. If box_size is a scalar, then the region shape will be
        (box_size, box_size). Either box_size or footprint must be defined.
        If they are both defined, then footprint overrides box_size.
    fwhm: float (Default: None)
        The full with at half maximum in unit of pixels. If not provided,
        it takes the size 1/8 of the box_size
    convolve: bool (Default: True)
        Set to True to first convovle the image with a Gaussian kernel of
        size sigma (next argument). This allows better centroiding of
        defocused images. This is ONLY USED for centroiding, the building of
        the PSF will be using the input data.
    sigma: float (Default: None)
        Defining the size of the Gaussian kernel, if not provided, it takes
        the value of fwhm/2.355.
    npeaks: int, optional (Default: 100)
        The maximum number of peaks to return. When the number of detected
        peaks exceeds npeaks, the peaks with the highest peak intensities
        will be returned.
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

    if threshold_snr is None:
        if convolve:
            # 3.0-sigma detection
            threshold_snr = 3.0

        else:
            # 5.0-sigma detection
            threshold_snr = 5.0

    else:
        threshold_snr = float(threshold_snr)

    if threshold is None:
        threshold = threshold_snr * np.nanstd(
            data[data < np.nanpercentile(data, threshold_clip_percentile)]
        )

    else:
        threshold = float(threshold)

    print(f"threshold is set at {threshold}.")

    if fwhm is None:
        fwhm = box_size / 8

    # Convolve with 2D gaussian to improve centroiding
    # First a 1-D  Gaussian
    if convolve:
        if sigma is None:
            sigma = fwhm / 2.355

        t = np.linspace(-10, 10, 30)
        bump = np.exp(-0.5 * (t / fwhm) ** 2)
        bump /= np.trapz(bump)  # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

        data_convolved = signal.fftconvolve(data, kernel, mode="same")

    else:
        data_convolved = data

    if stars_tbl is None:
        # detect peaks and remove sources near the edge
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, **kwargs)
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
        mask = distances < box_size
        # Ignore distnace to itself
        for i, _ in enumerate(_x):
            mask[i][i] = False

        mask = np.sum(mask, axis=0).astype("bool")
        peaks_tbl = peaks_tbl[~mask]

        x = peaks_tbl["xcentroid"]
        y = peaks_tbl["ycentroid"]

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

    nddata = NDData(data=data)

    # assign npeaks mask again because if stars_tbl is given, the npeaks
    # have to be selected
    stars = extract_stars(nddata, catalogs=stars_tbl[:npeaks], size=box_size)

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

    return stars, stars_tbl


def build_psf(
    stars,
    oversampling=None,
    return_oversampled=False,
    smoothing_kernel="quadratic",
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
    smoothing_kernel: {'quartic', 'quadratic'}, 2D ndarray, or None
                    (Default: quartic')
        The smoothing kernel to apply to the ePSF. The predefined 'quartic'
        and 'quadratic' kernels are derived from fourth and second degree
        polynomials, respectively. Alternatively, a custom 2D array can be
        input. If None then no smoothing will be performed.
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
        # Make sure the oversampling factor is sensible for the number
        # of stars available.
        if smoothing_kernel == "quartic":
            divisor = 4
        elif smoothing_kernel == "quadratic":
            divisor = 2
        else:
            divisor = 1
        oversampling = int(np.sqrt(len(stars) // divisor)) - 1
        oversampling = oversampling - oversampling % 2
        if oversampling < 2:
            oversampling = 2

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
