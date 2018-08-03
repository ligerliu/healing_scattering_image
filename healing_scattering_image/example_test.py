import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
plt.close('all')
from skimage.filters import median
from skimage.transform import resize
from heal_scatter_image import heal_scattering_image

im = np.double(np.load('example_2.npz')['im'])
mask = np.load('example_2.npz')['mask']

im[mask] = np.nan
xcenter = np.load('example_2.npz')['xcenter']
ycenter = np.load('example_2.npz')['ycenter']
r_min = 20
r_max = 2000#int(np.max(np.shape(im)))
angle_resolution = 360
delta_width = 2
bkgd_threshold  = 0.2
peaks_threshold = 4

healed_im, aniso_place, sym_record,\
iso_judge_global , iso_local_vs_global,\
qphi_image_3, I = heal_scattering_image(im,mask,
xcenter,
ycenter,
r_min,
r_max,
angle_resolution,
delta_width,
bkgd_threshold,
peaks_threshold,
bins = 30,
lambda_threshold_1 = 2.,
lambda_threshold_2 = 2.,
bkgd_fit_bias = 4.,
fold_bias = 4.,
fittness_threshold = 32,
extreme_fitness = 1e-3,
two_fold_apply = True,
fittness_prior=True,
down_sample = 0.1,)
