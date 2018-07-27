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

#iso_judge_global is iso_judge1 in old script
#aniso_judge is iso_judge in old script

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
                               hist_bins = 30):

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

# in old script anisotropic background No peaks was assigned aniso_bkgd = False
# diffuse = True, in new script this category was assigned as aniso_bkgd = True
# peaks = False, Thus judgement is little different from old ones
#

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
                        y1,
                        fold_bias,
                        fittness_threshold,
                        fittness_prior=True,
                        ):

    aniso_judge = ((iso_judge_global>
                    y[(lambda_global+lambda_threshold_1*
                    lambda_global**.5).astype(int)])
                    &((iso_local_vs_global)
                    <y1[(lambda_local_vs_global -
                    lambda_threshold_2*
                    lambda_local_vs_global**.5).astype(int)])).astype(int)

    aniso_span = consecutive(np.where(aniso_judge)[0])
    for i in range(len(aniso_span)):
        I1 = np.nansum(qphi_image_3[:,aniso_span[i]],axis=1)/len(aniso_span[i])
        qphi_aniso_span = np.copy(qphi_image_3[:,aniso_span[i]])
        if np.size(I1[np.isnan(I1)==0])<angle_resolution/8:
            II = (np.isnan(qphi_aniso_span)).astype(int)
            II = np.nansum(II,axis=1)
            II = (II>=len(aniso_span)/2)
            I1 = np.nanmean(qphi_aniso_span,axis=1)
            if np.size(II)<=(anlge_resolution-angle_resolution/16):
                I1[II] = np.NaN
        if (np.std(I1[isnan(I1) == 0])>
            np.std(np.nanmean(qphi_aniso_span,axis=1)
            [np.isnan(np.nanmean(qphi_aniso_span,axis=1))==0])):
            #II = (np.isnan(qphi_aniso_span)).astype(int)
            #II = np.sum(II,axis = 1)
            #II = (II>=len(aniso_span))
            #I1 = np.nanmean(qphi_aniso_span,axis=1)
            #I1[~II] = np.NaN
            I1 = np.nanmean(qphi_aniso_span,axis=1)
        I2 = np.copy(I1)
        I2 = I2 - np.nanmean(I1)
        # try to only emphasize the significant peaks which obvious above the average
        I2[I2 < 0] = 0
        I2[np.isnan(I2)] = 0
        # here correlation process will emphasize the symmertical information
        I3 = np.correlate(I2,I2,mode = 'smae')
        sym_fold = np.argsort(np.fft.rfft(I3)[2:10])[::-1]+2
        # we assume here only even order symmetry existed, means from 2 - 8 folds
        sym_fold = symfold[smy_fold%2 == 0]
        error_for_sym = np.zeros((len(sym_fold),))
        sym = 0
        fold_bias = fold_bias
        fittness_threshold = fittness_threshold

        for j in range(len(sym_fold)):
            model = np.nanmean(np.reshape(I1,(sym_fold[j],
                            angle_resolution/sym_fold[j])).T,axis=1)
            reshape_I1 = np.reshape(I1,(sym_fold[j],
                            angle_resolution/sym_fold[j])).T
            reshape_I1_nonnan = (np.isnan(reshape_I1)==0)
            num_nonnan_element_reshape = np.sum(reshape_I1_nonnan,axis=1)
            num_nonnan_repeat = np.size(num_nonnan_element_reshape[
                                 num_nonnan_element_reshape>1])
            error_for_sym = (np.nanmean(np.abs(I1-np.tile(model,sym_fold[j]))**2)
                            *(2-1.sym_fold[j]*num_nonnan_repeat/
                            angle_resolution)**fold_bias
        error_for_sym[error_for_sym == 0] = np.inf
        fittness_prior = True

        if (np.nanmean(iso_judge_global)/np.nanmean(I)/
        fittness_threshold>extreme_fitness):
            extreme_fitness = extreme_fitness
        else:
            extreme_fitness = (np.nanmean(iso_judge_global)/
            np.nanmean(I)/fittness_threshold

        if np.size(error_for_sym[error_for_sym<extreme_fitness]) > 1:
            sym = np.max(sym_fold[np.where(error_for_sym<extreme_fitness)[0]])
        else:
            if fittness_prior = True:
                sym = sym_fold[np.argmin(error_for_sym)]
            else:
                if (np.min(np.round(np.log(error_for_sym)))<=
                np.round(np.log(np.nanmean(iso_judge_global)/
                np.nanmean(I)/fittness_threshold))):
                if np.size(np.unique(np.log(error_for_sym))) > 1:
                    sym = np.max(sym_fold[
                    np.where(np.round(np.log(error_for_sym)) ==
                    np.min(np.unique(np.round(np.log(error_for_sym)))))[0]
                    ])

                    if np.size(np.unique(np.log(error_for_sym))) > 1:
                        sym = np.max(sym_fold[
                              np.where(np.round(np.log(error_for_sym)) ==
                              np.min(np.unique(np.round(np.log(error_for_sym))))
                              )[0]])
                        if (np.max(sym_fold[
                            np.where(np.floor(np.log(error_for_sym))==
                            np.min(np.unique(np.floor(np.log(error_for_sym)))))[0]
                            ])>sym):
                            sym = np.max(sym_fold[
                            np.where(np.floor(np.log(error_for_sym)) ==
                            np.min(np.unique(np.floor(np.log(error_for_sym)))))[0]
                            ])
                    else:
                        sym = np.max(sym_fold[:4])
                else:
                    sym = 0
        if (np.size(error_for_sym[error_for_sym == np.inf]) > 0) & \
           (np.size(error_for_sym[error_for_sym < extreme_fitness]) < 1):
           

plt.show()
