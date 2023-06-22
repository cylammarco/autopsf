import os

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astroscrappy import detect_cosmics
from photutils.psf import DAOPhotPSFPhotometry, EPSFModel

image_fits = fits.open(
    os.path.join("sbig_image", "coj1m011-kb05-20140430-0790-e90.fits.gz")
)[0]
image_data = detect_cosmics(image_fits.data, gain=1.7, readnoise=9.0)[1]
image_header = image_fits.header

psf_model_npy = np.load(
    os.path.join("sbig_image", "coj1m011-kb05-20140430-0790-e90_psf_model.npy")
)
psf_model = EPSFModel(psf_model_npy)

threshold = 10.0 * np.nanstd(
    image_data[image_data < np.nanpercentile(image_data, 80.0)]
)

fwhm = image_header["L1FWHM"]
pixel_scale = image_header["SECPIX"]

phot = DAOPhotPSFPhotometry(
    crit_separation=fwhm / pixel_scale,
    threshold=threshold,
    fwhm=fwhm,
    psf_model=psf_model,
    fitshape=len(psf_model_npy[0]),
    aperture_radius=fwhm / pixel_scale,
)
result_tab = phot(image_data)
residual_image = phot.get_residual_image()
