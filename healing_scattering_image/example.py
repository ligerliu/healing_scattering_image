import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
plt.close('all')
from skimage.filters import median

sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
2011_04Apr_30-Fang_shapedNPs/mini_image/\
lin_bs-CB_ring_top_20sec_frame9_SAXS.mat')

#sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
#2011_04Apr_30-Fang_shapedNPs/mini_image/\
#lin_bs-direct_60sec_SAXS.mat')

#sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
#2016_11Nov_17-First_samples/mini_image/\
#ChunliMa_2p7GPa_0004.mat')


im2 = loadmat(sample)['detector_image'].astype(float)
#im =ndimage.filters.median_filter(im2,size=5)
im = np.copy(im2)
im[175:180,:] = np.nan
im[230:235,:] = np.nan
im[93:225,124:132] = np.nan
r = [218,218,185,190,222,222]
c = [130,145,242,242,145,130]

rr,cc = polygon(r,c)
mask = np.zeros(np.shape(im)).astype(bool)
mask[im<10.] = True
mask[rr,cc] = True
mask[120:125,:] = True
mask[175:180,:] = True
mask[230:235,:] = True
mask[90:225,124:132] = True
mask[255,:] = True
im[mask] = np.nan

#fig,ax = plt.subplots()
#plt.imshow(np.log(im))

beam_x = 129
beam_y = 194

#im[im <= 5.] = np.nan

from skimage.transform import resize

from heal_scatter_image import heal_scattering_image

prefactor = 2
im = resize(im,(256*prefactor,256*prefactor))
mask = resize(mask,(256*prefactor,256*prefactor)).astype(bool)
xcenter = beam_x*prefactor
ycenter = beam_y*prefactor
r_min = 5*prefactor
r_max = 250*prefactor
angle_resolution = 360
delta_width = 2
bkgd_threshold  = 0.3
peaks_threshold = 4

healed_im, aniso_place, sym_record,\
iso_judge_global,iso_local_vs_global,\
qphi_image_3,I = heal_scattering_image(im,mask,
xcenter,
ycenter,
r_min,
r_max,
angle_resolution,
delta_width,
bkgd_threshold,
peaks_threshold,
bins = 30,
lambda_threshold_1 = 1.,
lambda_threshold_2 = 1.,
bkgd_fit_bias = 4.,
fold_bias = 4.,
fittness_threshold = 32,
extreme_fitness = 1e-3,
two_fold_apply = True,
fittness_prior=True,
down_sample = 0.1,)

'''
qphi_image_1,qphi_image_2,qphi_image_3 = calculate_qphi_image(im,
                         xcenter,
                         ycenter,
                         r_min,
                         r_max,
                         angle_resolution
                         )
I = oneD_intensity(im=im,xcenter=xcenter,ycenter=ycenter,mask=mask).cir_ave()
#fig,ax = plt.subplots()
#ax.plot(I)
I = I[:r_max]
iso_judge_global,iso_local_vs_global = two_iso_judge(
                  r_max,
                  qphi_image_3,
                  delta_width,
                  I )
#iso_judge_global is iso_judge1 in old script
#aniso_judge is iso_judge in old script
aniso_bkgd_judge,peaks_judge,\
lambda_global,\
lambda_local_vs_global,y,y1 = determine_scattering_class(iso_judge_global,
                               iso_local_vs_global,
                               bkgd_threshold,
                               peaks_threshold,
                               hist_bins)
'''
plt.show()
