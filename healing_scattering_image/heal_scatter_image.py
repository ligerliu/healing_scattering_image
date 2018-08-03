import numpy as np
import matplotlib.pyplot as plt # for showing image
import os
import sys
from scipy import signal, ndimage
from skimage import transform
from skimage.filters import rank
from oneD_intensity import oneD_intensity

def polar_coord(im,xcenter,ycenter,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(np.shape(im))
    #qphi_image=np.zeros((bins,max(shape_ind)))
    x_axis = np.arange(1,shape_ind[1]+1,1)#this is obtain from experimental setup
    y_axis = np.arange(1,shape_ind[0]+1,1)
    xcenter = float(xcenter)   # x coordinate of scattering center
    ycenter = float(ycenter) # y coordinate of scattering center


    xx , yy = np.meshgrid(x_axis, y_axis)
    xx = xx - xcenter
    # produce the x-coord for matrix with origin at scattering center
    yy = yy - ycenter
    # produce the y-coord for matrix with origin at scattering center
    azimuth = np.arctan(yy/xx)
    # arctan was specific for the N-D array calculation, atan only work for 1-D array
    azimuth = azimuth  + (xx<0)*np.pi +((xx>=0)&(yy<0))*2*np.pi
    radius = np.sqrt(xx**2+yy**2)
    radius = np.round(radius)
    radius = radius.astype(int)
    angle = np.round(np.round(azimuth*(bins/360)/np.pi*180)).astype(int)
    return radius,angle

def polar_coord_float(im,xcenter,ycenter,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(np.shape(im))
    #qphi_image=np.zeros((bins,max(shape_ind)))
    x_axis = np.arange(1,shape_ind[1]+1,1)#this is obtain from experimental setup
    y_axis = np.arange(1,shape_ind[0]+1,1)
    xcenter = float(xcenter)   # x coordinate of scattering center
    ycenter = float(ycenter) # y coordinate of scattering center


    xx , yy = np.meshgrid(x_axis, y_axis)
    xx = xx - xcenter
    # produce the x-coord for matrix with origin at scattering center
    yy = yy - ycenter
    # produce the y-coord for matrix with origin at scattering center
    azimuth = np.arctan(yy/xx)
    # arctan was specific for the N-D array calculation, atan only work for 1-D array
    azimuth = azimuth  + (xx<0)*np.pi +((xx>=0)&(yy<0))*2*np.pi
    radius = np.sqrt(xx**2+yy**2)
    angle = azimuth*(bins/360)/np.pi*180
    return radius,angle

def qphi_image(im,xcenter,ycenter,r_min,r_max,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(np.shape(im))
    qphi_image=np.zeros((bins,r_max))
    x_axis = np.arange(1,shape_ind[1]+1,1)#this is obtain from experimental setup
    y_axis = np.arange(1,shape_ind[0]+1,1)
    xcenter = float(xcenter)   # x coordinate of scattering center
    ycenter = float(ycenter) # y coordinate of scattering center


    xx , yy = np.meshgrid(x_axis, y_axis)
    xx = xx - xcenter
    # produce the x-coord for matrix with origin at scattering center
    yy = yy - ycenter
    # produce the y-coord for matrix with origin at scattering center
    azimuth = np.arctan(yy/xx)
    # arctan was specific for the N-D array calculation, atan only work for 1-D array
    azimuth = azimuth  + (xx<0)*np.pi +((xx>=0)&(yy<0))*2*np.pi
    radius = np.sqrt(xx**2+yy**2)
    radius = np.round(radius)
    radius = radius.astype(int)
    angle = np.round(np.round(azimuth*(bins/360)/np.pi*180)).astype(int)
    for i in range(r_min,r_max):
        azi_int = np.zeros((bins,))
        azi_int = np.bincount(angle[radius==i],weights=im[radius==i])
        azi_int = azi_int/np.bincount(angle[radius==i])
        if len(azi_int)>bins:
            azi_int=azi_int[:bins]
        elif  len(azi_int)<bins :
            azi_int=np.concatenate((azi_int,
                                    np.zeros((bins-len(azi_int),))*np.nan))
        qphi_image[:,i]=azi_int
    return qphi_image


def apply_two_fold_symmetry(im,xcenter,ycenter):

    imul = np.zeros((int(np.round(np.max([np.shape(im)[0]-ycenter,ycenter]))*2),
                     int(np.round(np.max([np.shape(im)[1]-xcenter,xcenter]))*2)
                     ))*np.nan
    imul[:np.shape(im)[0],:np.shape(im)[1]] = np.copy(im)
    imdr = np.flipud(np.fliplr(imul))
    shift_index_c=0
    shift_index_r=0
    shift_fitness= np.inf
    for shift_c in np.arange(-3,3,1):
        for shift_r in np.arange(-3,3,1):
            if shift_fitness>np.nanmean(np.abs(imul-\
               np.roll(np.roll(imdr,shift_c,axis=-1),shift_r,axis=0))):
               shift_fitness=np.nanmean(np.abs(imul-\
                             np.roll(np.roll(imdr,shift_c,axis=-1),\
                             shift_r,axis=0)))
               shift_index_c=int(shift_c)
               shift_index_r=int(shift_r)
    imdr = np.roll(np.roll(imdr,shift_index_c,axis=-1),shift_index_r,axis=0)
    imul[np.isnan(imul)] = imdr[np.isnan(imul)]
    im=np.copy(imul[:np.shape(im)[0],:np.shape(im)[1]])
    return im

def calculate_qphi_image(im,
                         xcenter,
                         ycenter,
                         r_min,
                         r_max,
                         angle_resolution
                         ):
    qphi_image_1 = qphi_image(im,xcenter,ycenter,r_min,r_max,angle_resolution)
    qphi_image_1[np.isnan(qphi_image_1)] = 1e-6
    qphi_image_1[qphi_image_1 <= 1] = 1e-6

    qphi_transform_mask = qphi_image(np.ones(np.shape(im)),
                                xcenter,ycenter,r_min,r_max,angle_resolution)
    #keep the transform mask
    qphi_image_1[np.isnan(qphi_transform_mask)] = np.nan
    #the mask correlates to the
    qphi_image_mask = (qphi_image_1 == 1e-6)

    #cut qphi_image_1 to r_max portion and make intensity to log scale which will make intensity comparable in variation scale
    #rotate to avoid mask at edge of qphi_image because mask at horizon or vertical to center
    qphi_image_2 = np.log(np.roll(qphi_image_1[:,:r_max],90,axis=0))
    qphi_image_2[qphi_image_2 < -6] = -6
    qphi_image_2_mask = (qphi_image_2 == -6)
    qphi_image_2_maks = ndimage.binary_dilation(qphi_image_2_mask,np.ones((2,2)))
    #dilute the mask ensure correct cover the edge of mask
    qphi_image_2[qphi_image_2_mask] = -6
    #feel like qphi_image_3 is redundant
    qphi_image_3 = np.copy(qphi_image_2)
    qphi_image_3[qphi_image_3 == -6] = np.nan
    qphi_3_mask = np.isnan(qphi_image_3)
    qphi_image_3_mask = ndimage.binary_dilation(qphi_3_mask,np.ones((5,5)))
    qphi_image_3_mask[:int(r_max/2),:] = False
    qphi_image_3[qphi_image_3_mask] = np.nan
    return qphi_image_1,qphi_image_2,qphi_image_3
#qphi_image_1[qphi_image_1==1e-6] = np.nan

def two_iso_judge(
                  r_max,
                  qphi_image_3,
                  delta_width,
                  angle_resolution,
                  I):
    iso_judge     = np.zeros((len(I),))
    iso_std                   = np.zeros((len(I),))
    iso_local_std             = np.zeros((len(I),))
    iso_judge_global          = np.zeros((len(I),))
    iso_local_vs_global = np.zeros((len(I),))
    for _ in range(r_max):
        if (np.size(qphi_image_3[np.isnan(qphi_image_3[:,_])==0,_]) >
           angle_resolution/10):
           iso_judge_global[_] = np.std(qphi_image_3[
                                 np.isnan(qphi_image_3[:,_]) == 0,_])
        else:
           iso_judge_global[_] = 0#np.nan

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

        Ave_I[ Ave_I < np.nanmean(Ave_I) - 2*np.std(np.nanmean(Ave_I))] = np.nan
        Ave_I_mean = np.concatenate((Ave_I[np.isnan(Ave_I)==0],
                                     Ave_I[np.isnan(Ave_I)==0],
                                     Ave_I[np.isnan(Ave_I)==0]))
        Ave_I_local_std = np.zeros((len(Ave_I[np.isnan(Ave_I)==0]),))
        len_nonnan_Ave_I = len(Ave_I[np.isnan(Ave_I)==0])
        for __ in range(len_nonnan_Ave_I,2*len_nonnan_Ave_I):
            Ave_I_local_std[__-len_nonnan_Ave_I] = np.std(
                                                   Ave_I_mean[
                                                   (__-int(angle_resolution/100)):
                                                   (__+int(angle_resolution/100))])

        iso_local_std[_] = np.abs(np.nanmean(Ave_I_local_std))
        iso_std[_]  = np.std(Ave_I_mean[len_nonnan_Ave_I:2*len_nonnan_Ave_I])
        iso_local_vs_global[_] = iso_local_std[_] / iso_std[_]
    return iso_judge_global,iso_local_vs_global


from scipy.misc import factorial
from scipy.optimize import curve_fit
def poisson_func(k,Lambda):
    return ((Lambda*1.)**k/factorial(k))*np.exp(-Lambda)

def determine_scattering_class(iso_judge_global,
                               iso_local_vs_global,
                               bkgd_threshold,
                               peaks_threshold,
                               hist_bins):

    # y is the
    x,y = np.histogram(iso_judge_global[np.isnan(iso_judge_global)==0],
                        bins=hist_bins)
    #make x,y elements corresponisible
    y = y[:-1] + np.diff(y)/2
    k = np.arange(1,hist_bins+1,1)

    if (np.size(x[np.isnan(x)]) > 0) or (np.size(x[x==0]) == len(x)):
       #return np.zeors(np.shape(im))
       print('break')
       return

    lambda_global = curve_fit(poisson_func,k,x*1./np.nansum(x),
                              bounds = [0,np.inf])[0][0]
    print(lambda_global)
    # y1 is the histogram of
    x1,y1 = np.histogram(iso_local_vs_global[
                    (np.abs(iso_local_vs_global) != np.inf) &
                    (np.isnan(iso_local_vs_global) == 0)],
                    bins = hist_bins)
    y1 = y1[1:] + np.diff(y1)/2.
    lambda_local_vs_global = curve_fit(poisson_func,k,x1*1./np.nansum(x1),\
                            bounds = [np.argmax(x1)-1,np.argmax(x1)+1])[0][0]
    print(np.abs(1-y1[lambda_local_vs_global.astype(int)]),lambda_global)
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
                           iso_judge_global,
                           iso_local_vs_global,
                           lamda_gloabl,
                           lambda_local_vs_global,
                           lambda_threshold_1,
                           radius,
                           r_min,
                           r_max,
                           y):
    #aniso_judge = np.zeros((len(I),))
    aniso_judge = ((iso_judge_global>
                    y[(lambda_global+lambda_threshold_1*\
                    lambda_global**.5).astype(int)])
                    &((iso_local_vs_global)<0.8)).astype(int)
    for _ in range(r_min,r_max):
        I_radius = im[radius == _]
        I_radius = I[_]
        im[radius == _] = IR

    return im,aniso_judge

def consecutive(data):
    return np.split(data.T, np.where(np.diff(data) != 1)[0] +1)

def heal_iso_bkgd_peaks(im,
                        I,
                        angle_resolution,
                        iso_judge_global,
                        iso_local_vs_global,
                        lambda_global,
                        lambda_local_vs_global,
                        lambda_threshold_1,
                        lambda_threshold_2,
                        radius,
                        a_f,
                        r_min,
                        r_max,
                        y,
                        y1,
                        fold_bias,
                        fittness_threshold,
                        fittness_prior,
                        qphi_image_3,
                        peaks_judge,
                        extreme_fitness,
                        fitting_shift
                        ):
    aniso_judge = ((iso_judge_global>
                    y[int(lambda_global+lambda_threshold_1*
                    lambda_global**.5)])
                    &((iso_local_vs_global)
                    <y1[int(lambda_local_vs_global -
                    lambda_threshold_2*
                    lambda_local_vs_global**.5)])).astype(int)

    aniso_span = consecutive(np.where(aniso_judge)[0])
    sym_record = np.zeros((len(aniso_span),))
    for i in range(len(aniso_span)):
        I1 = np.nansum(qphi_image_3[:,aniso_span[i]],axis=1)/len(aniso_span[i])
        qphi_aniso_span = np.copy(qphi_image_3[:,aniso_span[i]])
        if np.size(I1[np.isnan(I1)==0])<angle_resolution/8:
            II = (np.isnan(qphi_aniso_span)).astype(int)
            II = np.nansum(II,axis=1)
            II = (II>=len(aniso_span)/2)
            I1 = np.nanmean(qphi_aniso_span,axis=1)
            if np.size(II)<=(angle_resolution-angle_resolution/16):
                I1[II] = np.NaN
        if (np.std(I1[np.isnan(I1) == 0])>
            np.std(np.nanmean(qphi_aniso_span,axis=1)
            [np.isnan(np.nanmean(qphi_aniso_span,axis=1))==0])):
            II = (np.isnan(qphi_aniso_span)).astype(int)
            II = np.sum(II,axis = 1)
            II = (II>=len(aniso_span))
            I1 = np.nanmean(qphi_aniso_span,axis=1)
            I1[~II] = np.NaN
            I1 = np.nanmean(qphi_aniso_span,axis=1)
        I2 = np.copy(I1)
        I2 = I2 - np.nanmean(I1)
        # try to only emphasize the significant peaks which obvious above the average
        I2[I2 < 0] = 0
        I2[np.isnan(I2)] = 0
        # here correlation process will emphasize the symmertical information
        I3 = np.correlate(I2,I2,mode = 'same')
        sym_fold = np.argsort(np.fft.rfft(I3)[2:10])[::-1]+2
        # we assume here only even order symmetry existed, means from 2 - 8 folds
        sym_fold = sym_fold[sym_fold%2 == 0]
        error_for_sym = np.zeros((len(sym_fold),))
        sym = 0
        fold_bias = fold_bias
        fittness_threshold = fittness_threshold
        #I1 = I1.astype(np.float)
        #print(angle_resolution,sym_fold,I1.dtype,angle_resolution/sym_fold[0])
        for j in range(len(sym_fold)):
            model = np.nanmean(np.reshape(I1,(int(sym_fold[j]),
                            int(angle_resolution/sym_fold[j]))).T,axis=1)
            reshape_I1 = np.reshape(I1,(int(sym_fold[j]),
                            int(angle_resolution/sym_fold[j]))).T
            reshape_I1_nonnan = (np.isnan(reshape_I1)==0)
            num_nonnan_element_reshape = np.sum(reshape_I1_nonnan,axis=1)
            num_nonnan_repeat = np.size(num_nonnan_element_reshape[
                                 num_nonnan_element_reshape>1])
            error_for_sym[j] = (np.nanmean(np.abs(I1-np.tile(model,sym_fold[j]))**2)
                            *(2-1.*sym_fold[j]*num_nonnan_repeat/
                            angle_resolution)**fold_bias)
        #print(error_for_sym,I1)
        error_for_sym[error_for_sym == 0] = np.inf
        fittness_prior = True

        if (np.nanmean(iso_judge_global)/np.nanmean(I)/
            fittness_threshold>extreme_fitness):
            extreme_fitness = extreme_fitness
        else:
            extreme_fitness = (np.nanmean(iso_judge_global)/
            np.nanmean(I)/fittness_threshold)

        if np.size(error_for_sym[error_for_sym<extreme_fitness]) > 1:
            sym = np.max(sym_fold[np.where(error_for_sym<extreme_fitness)[0]])
        else:
            if fittness_prior == True:
                sym = sym_fold[np.argmin(error_for_sym)]
            else:
                if (np.min(np.round(np.log(error_for_sym)))<=
                    np.round(np.log(np.nanmean(iso_judge_global)/
                    np.nanmean(I)/fittness_threshold))):
                    if np.size(np.unique(np.log(error_for_sym))) > 1:
                        sym = np.max(sym_fold[
                              np.where(np.round(np.log(error_for_sym)) ==
                              np.min(np.unique(np.round(
                              np.log(error_for_sym)))))[0]
                               ])
                        if np.size(np.unique(np.log(error_for_sym))) > 1:
                            sym = np.max(sym_fold[
                                  np.where(np.round(np.log(error_for_sym)) ==
                                  np.min(np.unique(
                                  np.round(np.log(error_for_sym))))
                                  )[0]])
                            if (np.max(sym_fold[
                                np.where(np.floor(np.log(error_for_sym))==
                                np.min(np.unique(np.floor(
                                np.log(error_for_sym)))))[0]
                                ])>sym):
                                sym = np.max(sym_fold[
                                np.where(np.floor(np.log(error_for_sym)) ==
                                np.min(np.unique(np.floor(
                                np.log(error_for_sym)))))[0]
                                ])
                        else:
                            sym = np.max(sym_fold[:4])
                else:
                    sym = 0

        # this could be have when there is no noise in the data,
        # then 4, 6, or 8 fold will auto have 2 fold sym,
        # we try to choose largest one.

        if (np.size(error_for_sym[error_for_sym == np.inf]) > 0) & \
           (np.size(error_for_sym[error_for_sym < extreme_fitness]) < 1):
           if (np.size(error_for_sym[error_for_sym==np.inf] ==1)) and \
              (np.min(np.round(np.log(error_for_sym))) > \
               np.round(np.log(np.nanmean(iso_judge_global)/np.nanmean(I)/\
               fittness_threshold))):
               sym = sym_fold[error_for_sym == np.inf][0]
           else:
               sym = np.max(sym_fold[np.where(error_for_sym==np.inf)[0]])

        # after know the symmtry, we try to build two D model by separate
        # qphi_image basing on sym folds
        if sym > 0:
            I6 = np.zeros((int(angle_resolution/sym),len(aniso_span[i]),sym))
            for k in range(sym):
                I6[:,:,k] = np.copy(qphi_image_3[int(k*angle_resolution/sym):
                                           int((k+1)*angle_resolution/sym),
                                           aniso_span[i]])
            nonnan_size = np.zeros((sym,))
            for n in range(sym):
                nonnan_size[n] = np.size(I6[np.isnan(I6[:,:,n])==0,n])
            TwoD_model = I6[:,:,np.argmax(nonnan_size)]
            sym_region_shift_r = np.zeros((sym,)).astype(int)
            sym_region_shift_c = np.zeros((sym,)).astype(int)
            if fitting_shift == True:
                for n in range(sym):
                    sym_region_fitness = np.inf
                    if n != np.argmax(nonnan_size):
                        for m in np.arange(-5,5,1):
                            for mm in np.arange(-5,5,1):
                                if sym_region_fitness > np.nanmean(
                                   np.abs(TwoD_model-(np.roll(np.roll(
                                   qphi_image_3,m,axis=-1),mm,axis=0))
                                   [int(n*angle_resolution/sym):
                                   int((n+1)*angle_resolution/sym),
                                   aniso_span[i]])):
                                    sym_region_fitness = np.nanmean(
                                       np.abs(TwoD_model - (np.roll(np.roll(
                                       qphi_image_3,m,axis=-1),mm,axis=0))
                                       [int(n*angle_resolution/sym):int((n+1)*
                                       angle_resolution/sym),aniso_span[i]]))
                                    sym_region_shift_c[n] = m
                                    sym_region_shift_r[n] = mm
                        I6[:,:,n] = np.roll(np.roll(qphi_image_3,
                                    sym_region_shift_c[n],axis=-1),
                                    sym_region_shift_r[n],axis=0)[
                                    int(n*angle_resolution/sym):
                                    int((n+1)*angle_resolution/sym),
                                    aniso_span[i]]
                if np.sum(np.sum(np.nanmean(I6,axis=2))) == np.nan:
                    if peaks_judge == True:
                        from skimage import restoration
                        TwoD_model = restoration.inpaint_biharmonic(
                        np.nanmean(I6,axis = 2),
                        mask = (np.isnan(np.nanmean(I6,axis=2)))
                        )
                    elif peaks_judge == False:
                        TwoD_model = np.nanmean( I6 , aixs = 2)
                        TwoD_healing = np.exp(np.copy(TwoD_model))
                        TwoD_healing[np.isnan(TwoD_healing)] = 0
                        TwoD_model[np.isnan(TwoD_model)] = 0
                        while np.size(TwoD_healing[TwoD_healing == 0]) > 0:
                            from skimage.filters import rank
                            TwoD_healing = rank.mean(TwoD_healing.astype(unit16),
                                            np.ones((5,5)),
                                            mask = (TwoD_healing != 0))
                            TwoD_model[TwoD_model == 0] = np.log((TwoD_healing
                                               [TwoD_model == 0]).astype(float))
                for k in range(sym):
                    fit_model = np.zeros(np.shape(TwoD_model))
                    fit_model = np.roll(np.roll(TwoD_model,
                                   -sym_region_shift_c[k],axis = -1),
                                   -sym_region_shift_r[k],axis = 0)
                    fit_model[:,:sym_region_shift_c[k]] = np.copy(
                                    TwoD_model[:,:sym_region_shift_c[k]]
                                    )
                    I6[:,:,k][np.isnan(I6[:,:,k])==1] = fit_model[
                                    np.isnan(I6[:,:,k])==1]
                    qphi_image_3[int(k*angle_resolution/sym):
                         int((k+1)*angle_resolution/sym)
                         ,aniso_span[i]] = I6[:,:,k]
        sym_record[i] = sym
    # qphi_image_6 is healed qphi_image in real intensity scale.
    qphi_image_6 = np.exp(np.roll(qphi_image_3,-90,axis=0))

    aniso_span1  = consecutive(np.where(aniso_judge)[0])
    aniso_area = aniso_span1[0]
    for s in range(1,len(aniso_span1)):
        aniso_area = np.append(aniso_area,aniso_span1[s])

    im2 = np.copy(im)
    for _ in range(r_min,r_max):
        if _ in aniso_area:
            IR = im2[radius == _]
            FR = np.interp(a_f[radius == _], np.arange(0,
                 angle_resolution,1)[np.isnan(qphi_image_6[:,_])
                 ==0],qphi_image_6[np.isnan(qphi_image_6[:,_])==0,_])

            IR[np.isnan(IR)] = FR[np.isnan(IR)]
            im2[radius == _] = IR
        else:
            IR = im2[radius==_]
            IR[np.isnan(IR)] = I[_]
            im2[radius == _] = IR

    return im2,aniso_judge,sym_record


def heal_scattering_image(
                          im,
                          mask,
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
                          fittness_threshold = 64,
                          extreme_fitness = 1e-3,
                          two_fold_apply = True,
                          fittness_prior=True,
                          down_sample = 0.1,
                          fitting_shift = True
                          ):

    radius,azimuth = polar_coord(im,xcenter,ycenter,angle_resolution)
    r_f,a_f = polar_coord_float(im,xcenter,ycenter,angle_resolution)

    if two_fold_apply == True:
        im = apply_two_fold_symmetry(im,xcenter,ycenter)
    else:
        im = np.copy(im)
    #im[im<1] = 1
    I = oneD_intensity(im=im,xcenter=xcenter,ycenter=ycenter,mask=mask).cir_ave()
    #fig,ax = plt.subplots()
    #ax.plot(I)
    I = I[:r_max]

    qphi_image_1,qphi_image_2,qphi_image_3 = calculate_qphi_image(im,
                                                             xcenter,
                                                             ycenter,
                                                             r_min,
                                                             r_max,
                                                             angle_resolution)

    iso_judge_global,iso_local_vs_global = two_iso_judge(r_max,
                                                         qphi_image_3,
                                                         delta_width,
                                                         angle_resolution,
                                                         I,)
    aniso_bkgd_judge,peaks_judge,\
    lambda_global,\
    lambda_local_vs_global,y,y1  = determine_scattering_class(iso_judge_global,
                                   iso_local_vs_global,
                                   bkgd_threshold,
                                   peaks_threshold,
                                   hist_bins = bins)
    #if lambda_global < 1:
    #    lambda_global = 1
    #if lambda_local_vs_global < 1:
    #    lambda_local_vs_global = 1
    #peaks_judge = True
    #aniso_bkgd_judge = False
    print(aniso_bkgd_judge,peaks_judge)
    if aniso_bkgd_judge == False and peaks_judge == False:
        im,aniso_judge = heal_iso_bkgd_no_peaks(I,
                                   lambda_global,
                                   lambda_local_vs_global,
                                   lambda_threshold_1,
                                   radius,
                                   r_min,
                                   r_max,
                                   y)
        sym_record = np.nan

    if aniso_bkgd_judge == False and peaks_judge == True:
        im,aniso_judge,sym_record = heal_iso_bkgd_peaks(im,
                                I,
                                angle_resolution,
                                iso_judge_global,
                                iso_local_vs_global,
                                lambda_global,
                                lambda_local_vs_global,
                                lambda_threshold_1,
                                lambda_threshold_2,
                                radius,
                                a_f,
                                r_min,
                                r_max,
                                y,
                                y1,
                                fold_bias,
                                fittness_threshold,
                                fittness_prior,
                                qphi_image_3,
                                peaks_judge,
                                extreme_fitness,
                                fitting_shift
                                )

    if aniso_bkgd_judge == True and peaks_judge == False:
        im,aniso_judge,sym_record = heal_iso_bkgd_peaks(im,
                                I,
                                angle_resolution,
                                iso_judge_global,
                                iso_local_vs_global,
                                lambda_global,
                                lambda_local_vs_global,
                                lambda_threshold_1,
                                lambda_threshold_2,
                                radius,
                                a_f,
                                r_min,
                                r_max,
                                y,
                                y1,
                                fold_bias,
                                fittness_threshold,
                                fittness_prior,
                                qphi_image_3,
                                peaks_judge,
                                extreme_fitness,
                                fitting_shift
                                )

    if aniso_bkgd_judge == True and peaks_judge == True:
        im = np.copy(im)
        sym_record = np.nan
        aniso_judge = ((iso_judge_global>
                            y[(lambda_global+lambda_threshold_1*
                            lambda_global**.5).astype(int)])
                            &((iso_local_vs_global)
                            <y1[(lambda_local_vs_global -
                            lambda_threshold_2*
                            lambda_local_vs_global**.5).astype(int)])).astype(int)
    return im, aniso_judge, sym_record ,iso_judge_global,iso_local_vs_global,\
    qphi_image_3,I
