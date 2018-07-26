import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
plt.close('all')

#sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
#2011_04Apr_30-Fang_shapedNPs/mini_image/\
#lin_bs-CB_ring_top_20sec_frame9_SAXS.mat')

sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
2011_04Apr_30-Fang_shapedNPs/mini_image/\
lin_bs-direct_60sec_SAXS.mat')

#sample = ('/Users/jiliangliu/Desktop/CMS_for_ML/\
#2016_11Nov_17-First_samples/mini_image/\
#ChunliMa_2p7GPa_0004.mat')
im = loadmat(sample)['detector_image'].astype(float)

im[175:180,:] = np.nan
im[230:235,:] = np.nan
im[93:223,124:132] = np.nan
r = [218,218,185,190,222,222]
c = [130,145,242,242,145,130]

rr,cc = polygon(r,c)
mask = np.zeros(np.shape(im)).astype(bool)
mask[rr,cc] = True
mask[175:180,:] = True
mask[230:235,:] = True
mask[93:223,124:132] = True

im[mask] = np.nan

fig,ax = plt.subplots()
plt.imshow(np.log(im))

beam_x = 129
beam_y = 194

im[im <= 0.] = 1e-6

from healing_scattering_image_packaging import polar_coord,polar_coord_float,qphi_image,oneD_intensity

qphi_image_1 = qphi_image(im,beam_x,beam_y,4,220,720)

fig,ax = plt.subplots()
plt.imshow(np.log(qphi_image_1))

xcenter = beam_x
ycenter = beam_y
r_min = 4
r_max = 220
angle_resolution = 720


qphi_image_1[np.isnan(qphi_image_1)] = 1e-6
qphi_image_1[qphi_image_1 < 1e-6] = 1e-6

qphi_transform_mask = qphi_image(np.ones(np.shape(im)),xcenter,ycenter,r_min,r_max,angle_resolution)
#keep the transform mask
qphi_image_1[np.isnan(qphi_transform_mask)] = np.nan
#the mask correlates to the 
qphi_image_mask = (qphi_image_1 == 1e-6)

#cut qphi_image_1 to r_max portion and make intensity to log scale which will make intensity comparable in variation scale
#rotate to avoid mask at edge of qphi_image because mask at horizon or vertical to center
qphi_image_2 = np.roll(qphi_image_1[:,:r_max],90,axis=0)
qphi_image_2[qphi_image_2 < -6] = -6
qphi_image_2_mask = (qphi_image_2 == -6)
qphi_image_2_maks = ndimage.binary_dilation(qphi_image_2_mask,np.ones((2,2)))
#dilute the mask ensure correct cover the edge of mask
qphi_image_2[qphi_image_2_mask] = -6
#feel like qphi_image_3 is redundant
qphi_image_3 = np.copy(qphi_image_2)
qphi_image_3[qphi_image_3 == -6] = np.nan
radius,azimuth = polar_coord(im,xcenter,ycenter,angle_resolution)
r_f,a_f = polar_coord_float(im,xcenter,ycenter,angle_resolution)

#im[im<1] = 1
I = oneD_intensity(im=im,xcenter=xcenter,ycenter=ycenter,mask=mask).cir_ave()
fig,ax = plt.subplots()
ax.plot(I)

I = I[:r_max]
iso_judge     = np.zeros((len(I),))

iso_std                   = np.zeros((len(I),))
iso_local_std             = np.zeros((len(I),))
iso_judge_global          = np.zeros((len(I),))
iso_judge_local_vs_global = np.zeros((len(I),))

delta_width = 1

qphi_image_1[qphi_image_1==1e-6] = np.nan


for _ in range(r_max):
    if (np.size(qphi_image_3[np.isnan(qphi_image_3[:,_])==0,_]) >
       angle_resolution/10):
       iso_judge_global[_] = np.std(qphi_image_3[
                             np.isnan(qphi_image_3[:,_]) == 0,_])
    else:
       iso_judge_global[_] = np.nan
    
    # here try to calculate the intenity in each q with smooth of delta_width of q
    if   (_ > delta_width) and (_ < len(I)-delta_width):
       Ave_I = np.sum(qphi_image_3[:, (_-delta_width) : (_+delta_width)],
                      axis = 1) / (2*delta_width)
    elif (_ < delta_width):
       Ave_I = np.sum(qphi_image_3[:, _ : (_+delta_width)],
                      axis = 1) / delta_width
    elif (_ > len(I) - delta_width):
       Ave_I = np.sum(qphi_image_3[:, (_-delta_width) : _],
                      axis = 1) / delta_width
    
    Ave_I[ Ave_I < np.nanmean(Ave_I) - 0.7*np.abs(np.nanmean(Ave_I))] = np.nan
    Ave_I_mean = np.concatenate((Ave_I[np.isnan(Ave_I)==0],
                                 Ave_I[np.isnan(Ave_I)==0],
                                 Ave_I[np.isnan(Ave_I)==0]))
    Ave_I_local_std = np.zeros((len(Ave_I[np.isnan(Ave_I)==0]),))
    len_nonnan_Ave_I = len(Ave_I[np.isnan(Ave_I)==0])
    for __ in range(len_nonnan_Ave_I,2*len_nonnan_Ave_I):
        Ave_I_local_std[__-len_nonnan_Ave_I] = np.std(
                                               Ave_I_mean[(__-int(angle_resolution/100)):
                                               (__+int(angle_resolution/100))])
        
    iso_local_std[_] = np.abs(np.nanmean(Ave_I_local_std))
    iso_std[_]  = np.std(Ave_I_mean[len_nonnan_Ave_I:2*len_nonnan_Ave_I])
    iso_judge_local_vs_global[_] = iso_local_std[_] / iso_std[_]
    

from scipy.misc import factorial
from scipy.optimize import curve_fit
def poisson_func(k,Lambda):
    return ((Lambda*1.)**k/factorial(k))*np.exp(-Lambda)

bkgd_threshold  = 0.2
peaks_threshold = 3
def determine_scattering_class(iso_judge_global,
                               iso_judge_local_vs_global,
                               bkgd_threshold,
                               peaks_threshold,
                               hist_bins = 30)
    
    # y is the 
    x,y = np.histogram(iso_judge_global[np.isnan(iso_judge_global)==0] ,bins=hist_bins)
    #make x,y elements corresponisible
    y = y[:-1] + np.diff(y)/2
    k = np.arange(1,hist_bins+1,1)
    
    if (np.size(x[np.isnan(x)]) > 0) or (np.size(x[x==0]) == len(x)):
       #return np.zeors(np.shape(im))
       print('break')
       break
    
    lambda_global = curve_fit(poisson_func,k,x*1./np.nansum(x),bounds = [0,np.inf])[0][0]
    # y1 is the histogram of 
    x1,y1 = np.histogram(iso_judge_local_vs_global[
                    (np.abs(iso_judge_local_vs_global) != np.inf) &
                    (np.isnan(iso_judge_local_vs_global) == 0)], 
                    bins = hist_bins)
    y1 = y1[1:] + np.diff(y1)/2.
    lambda_local_vs_global = curve_fit(poisson_func,k,x1*1./np.nansum(x1),\
                            bounds = [np.argmax(x1)-1,np.argmax(x1)+1])[0][0]
    
    if (np.abs(1-y1[lambda_local_vs_global.astype(int)]) < bkgd_threshold):
       aniso_bkgd_judge = False
       if lambda_global <= peaks_threshold:
          peaks_judge = True
       else:
          peaks_judge = False
    else:
       aniso_bkgd_judge = True
       if lambda_global <= peaks_threshold:
          peaks_judge = True
       else:
          peaks_judge = False
    return aniso_bkgd_judge,peaks_judge,lambda_global,lambda_local_vs_global,y,y1

def heal_iso_bkgd_no_peaks(I,
                           lamda_gloabl,
                           lambda_local_vs_global,
                           lambda_threshold_1 = 2.
                           radius,
                           r_min = 0,
                           r_max,
                           y
                           ):
    
    aniso_judge = ((iso_judge_global>
                    y[(lambda_global+lambda_threshold_1*lambda_global**.5).astype(int)])
                    &((iso_local_vs_global)<0.8)).astype(int)
    for _ in range(r_min,r_max):
        I_radius = im[radius == _]
        I_radius = I[_]
        im[radius == _] = IR
    
    return im,aniso_judge
def consecutive(data):
    return np.split(data.T, np.where(np.diff(data) != 1)[0] +1)
def heal_iso_bkgd_peaks(I,
                        lambda_global,
                        lambda_local_vs_global,
                        lambda_threshold_1 = 2.,
                        lambda_threshold_2 = 2.,
                        radius,
                        r_min = 0,
                        r_max,
                        y,
                        y1
                        ):
    
    aniso_judge = ((iso_judge_global>
                    y[(lambda_global+lambda_threshold_1*lambda_global**.5).astype(int)])
                    &((iso_local_vs_global)
                    <y1[(lambda_local_vs_global -
                    lambda_threshold_2*lambda_local_vs_global**.5).astype(int)])).astype(int)
    
plt.show()
