from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *

# Define the ROI
ra_crab, dec_crab = 83.63,22.02
data_radius = 3.0
model_radius = 8.0

roi = HealpixConeROI(data_radius=data_radius,
                     model_radius=model_radius,
                     ra=ra_crab,
                     dec=dec_crab)

# Instance the plugin
maptree = "../data/HAWC_9bin_507days_crab_data.hd5"
response = "../data/HAWC_9bin_507days_crab_response.hd5"

hawc = HAL("HAWC",
           maptree,
           response,
           roi,
           flat_sky_pixels_size=0.1)

# Use from bin 1 to bin 9
hawc.set_active_measurements(1, 9)

# Display information about the data loaded and the ROI
hawc.display()

# Look at the data
fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.17)

# Save to file
fig.savefig("public_crab_logParabola_stacked_image.png")


# Define model
spectrum = Log_parabola()
source = PointSource("crab", ra=ra_crab, dec=dec_crab, spectral_shape=spectrum)

spectrum.piv = 7 * u.TeV
spectrum.piv.fix = True

spectrum.K = 1e-14 / (u.TeV * u.cm ** 2 * u.s)  # norm (in 1/(keV cm2 s))
spectrum.K.bounds = (1e-35, 1e-10) / (u.TeV * u.cm ** 2 * u.s)  # without units energies are in keV


spectrum.alpha = -2.5  # log parabolic alpha (index)
spectrum.alpha.bounds = (-4., 2.)

spectrum.beta = 0  # log parabolic alpha (index)
spectrum.beta.bounds = (-4., 2.)

model = Model(source)

data = DataList(hawc)

jl = JointLikelihood(model, data, verbose=False)
jl.set_minimizer("minuit")
param_df, like_df = jl.fit()

results=jl.results
results.write_to("crab_lp_public_results.fits", overwrite=True)
results.optimized_model.save("crab_fit.yml", overwrite=True)

# See the model in counts space and the residuals
fig = hawc.display_spectrum()

# Save it to file
fig.savefig("public_crab_logParabola_residuals.png")

# See the spectrum fit
fig = plot_spectra(jl.results,
                   ene_min=1.0,
                   ene_max=37,
                   num_ene=50,
                   energy_unit='TeV',
                   flux_unit='TeV/(s cm2)')
plt.xlim(0.8,100)
plt.ylabel(r"$E^2\,dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]")
plt.xlabel("Energy [TeV]")
fig.savefig("public_crab_fit_spectrum.png")

# Look at the different energy planes (the columns are model, data, residuals)
fig = hawc.display_fit(smoothing_kernel_sigma=0.3,display_colorbar=True)
fig.savefig("public_crab_fit_planes.png")

# Compute TS
TS = jl.compute_TS("crab", like_df)
print(TS)
