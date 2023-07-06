sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MMMBackground()
bkg = Background2D(
    image_data,
    (51, 51),
    filter_size=(15, 15),
    sigma_clip=sigma_clip,
    bkg_estimator=bkg_estimator,
)

bkg_norm = simple_norm(bkg.background, "log", percent=95.0)
img_norm = simple_norm(image_data, "log", percent=95.0)
# save the background
plt.figure(1, (10, 10))
plt.clf()
plt.imshow(bkg.background, origin="lower", aspect="auto", norm=bkg_norm)


from astropy.visualization import simple_norm
from matplotlib.pyplot import *
from PyZOGY.subtract import run_subtraction

ion()


diff = run_subtraction(
    science_image=r"C:\Users\cylam\git\autopsf\examples\sinistro_image\tfn1m014-fa20-20221201-0087-e91_background_subtracted.fits",
    science_psf=r"C:\Users\cylam\git\autopsf\examples\sinistro_image\tfn1m014-fa20-20221201-0087-e91_psf_model.fits",
    reference_image=r"C:\Users\cylam\git\autopsf\examples\sinistro_image\tfn1m014-fa20-20221201-0088-e91_background_subtracted.fits",
    reference_psf=r"C:\Users\cylam\git\autopsf\examples\sinistro_image\tfn1m014-fa20-20221201-0088-e91_psf_model.fits",
    # reference_mask = refdata.mask,
    gain_ratio=1.0,
    n_stamps=1,
    science_saturation=100000,
    reference_saturation=100000,
    normalization="science",
    sigma_cut=10.0,
    photometry=True,
    show=True
    # size_cut = True, sigma_cut=3
)

norm_diff = simple_norm(diff[0], "log", percent=95.0)
figure(4)
clf()
imshow(diff[0], aspect="auto", norm=norm_diff)
