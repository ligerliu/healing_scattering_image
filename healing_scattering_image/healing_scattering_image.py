#import time
import numpy as np
import matplotlib.pyplot as plt # for showing image
from pylab import * # for multiple figure window
import os
#from lmfit import Model
import sys
#from lmfit.models import GaussianModel, ExponentialModel, LorentzianModel
from scipy import signal, ndimage
from skimage import transform
from skimage.filters import rank
#from skimage.restoration import denoise_tv_chambolle

#t = time.time()
'''
address = '/Users/jiliangliu/Desktop/synthetic/data_fullsize/'
os.chdir(address)

q=4*np.pi*np.sin(np.arctan(np.arange(1,2168,1)*75*1e-6/4.9)/2)/1.4234
a = np.load('test1.npz')
intensity=a['intensity']
sample_name = a['sample_name']
x_center=a['x_center']
y_center=a['y_center']

i = 10968#10813#966#10919#10963#10888# the problem of 11233 and 10963 is too large span blur the symmetry features
xcenter = float(x_center[i])   # x coordinate of scattering center
ycenter = float(y_center[i]) 

im = double(load(sample_name[i]))

'''
'''
address = '/Users/jiliangliu/Dropbox/Aug_18/sample'

os.chdir(address)

im=load('image_artifact.npz')['im']
im=double(im)
xcenter=1209.115   
ycenter=1328.118

'''
'''
im=denoise_tv_chambolle(im,weight=10)
#im=ndimage.filters.median_filter(im,size=6)
#zinger_threshold = 3*3
#zinger_mask = zinger_threshold*double(ndimage.filters.median_filter(im,size=6))
#mask = (im>60000)  | ((im-zinger_mask)>0) #| (im<5)
mask = (im>600000)
#mask=np.zeros(shape(im))
#mask=mask.astype(bool)
mask[:,1028:1043]=True
mask[512:553,]=True;mask[252:259,]=True;
mask[803:812,]=True;mask[1613:1656,]=True;mask[1062:1104,]=True;mask[1354:1364,]=True;mask[1906:1914,]=True

#mask[1320:1336,1184:]=True
##mask[1300:,1171:1251]=True
#from skimage.draw import polygon
#r = np.array([1300,2167,2167,1300])
#c = np.array([1171,1171,1300,1251])
#rr,cc=polygon(r,c)
#mask[rr,cc]=True

mask[1871:,720:785]=True

#from skimage.draw import polygon
#r = np.array([1728,2167,2167,1728])
#c = np.array([1140,1200,1407,1158])
#rr,cc=polygon(r,c)
#mask[rr,cc]=True

#mask=ndimage.binary_dilation(mask,np.ones((5,5)))
im[mask]=np.nan
#figure(11),imshow(log(im))#imshow(log(im),vmin=-3,vmax=6)

'''

def polar_coord(im,xcenter,ycenter,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(shape(im))
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
    azimuth = azimuth  + (xx<0)*pi +((xx>=0)&(yy<0))*2*pi
    radius = np.sqrt(xx**2+yy**2)
    radius = np.round(radius)
    radius = radius.astype(int)
    angle = np.round(np.round(azimuth*(bins/360)/pi*180)).astype(int)
    return radius,angle 
    
def polar_coord_float(im,xcenter,ycenter,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(shape(im))
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
    azimuth = azimuth  + (xx<0)*pi +((xx>=0)&(yy<0))*2*pi
    radius = np.sqrt(xx**2+yy**2)
    angle = azimuth*(bins/360)/pi*180
    return radius,angle

def qphi_image(im,xcenter,ycenter,r_min,r_max,bins):
    #should be able to change 360 to 720 by change the delta phi
    shape_ind = np.asarray(shape(im))
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
    azimuth = azimuth  + (xx<0)*pi +((xx>=0)&(yy<0))*2*pi
    radius = np.sqrt(xx**2+yy**2)
    radius = np.round(radius)
    radius = radius.astype(int)
    angle = np.round(np.round(azimuth*(bins/360)/pi*180)).astype(int)
    for i in range(r_min,r_max):
        azi_int = np.zeros((bins,))
        azi_int = np.bincount(angle[radius==i],weights=im[radius==i])
        azi_int = azi_int/np.bincount(angle[radius==i])
        if len(azi_int)>bins:
            azi_int=azi_int[:bins]
        elif  len(azi_int)<bins :
            azi_int=np.concatenate((azi_int,np.zeros((bins-len(azi_int),))*nan))
        qphi_image[:,i]=azi_int
    return qphi_image
   
def image_healing(im,xcenter,ycenter,mask,r_max,r_min,angle_resolution=720,bkdg_fit_bias=4,fold_bias=4,fittness_threshold=64,extreme_fitness=1e-3,fittness_porior=True,down_sample=.1,lamda_threshold_1=1.,lamda_threshold_2=1.,lamda_threshold_3=1.,bkgd_threshold=.2,fitting_shift=True,two_fold_apply =True):
    #angle_resolution=720
    #r_max=2000#np.max(shape(im))
    #r_min=3
    if two_fold_apply ==True:
        imul = np.zeros((np.round(np.max([shape(im)[0]-ycenter,ycenter]))*2,np.round(np.max([shape(im)[1]-xcenter,xcenter]))*2))*np.nan
        imul[:shape(im)[0],:shape(im)[1]] = np.copy(im)
        imdr = flipud(fliplr(imul))
        shift_index_c=0
        shift_index_r=0
        shift_fitness=inf
        for shift_c in np.arange(-3,3,1):
            for shift_r in np.arange(-3,3,1):
                if shift_fitness>nanmean(np.abs(imul-np.roll(np.roll(imdr,shift_c,axis=-1),shift_r,axis=0))):
                    shift_fitness=nanmean(np.abs(imul-np.roll(np.roll(imdr,shift_c,axis=-1),shift_r,axis=0)))
                    shift_index_c=int(shift_c)
                    shift_index_r=int(shift_r)
        imdr = np.roll(np.roll(imdr,shift_index_c,axis=-1),shift_index_r,axis=0)
        imul[isnan(imul)] = imdr[isnan(imul)]
        im=np.copy(imul[:shape(im)[0],:shape(im)[1]])
    else:
        im = np.copy(im)
    ###two fold application
    qphi_image1=qphi_image(im,xcenter,ycenter,r_min,r_max,angle_resolution)
    qphi_image1[isnan(qphi_image1)]=1e-6
    qphi_mask=qphi_image(np.ones((shape(im))),xcenter,ycenter,r_min,r_max,angle_resolution)
    qphi_image1[isnan(qphi_mask)]=np.nan
    qphi_mask2=(qphi_image1==1e-6)
    #print time.time()-t
    #np.roll could apply on 2D image, 
    #but skimage inpaint is not very well perfomred on x-ray data reconstruction
    qphi_image1[qphi_image1==0]=1e-6
    qphi_image1[qphi_image1<1e-6]=1e-6
    qphi_image2=log(np.roll(qphi_image1[:,0:r_max],90,axis=0))
    qphi_image2[qphi_image2<-6]=-6
    mask2=(qphi_image2==-6)
    mask2=ndimage.binary_dilation(mask2,np.ones((2,2)))
    qphi_image2[mask2]=-6
    qphi_image3=np.copy(qphi_image2)
    qphi_image3[qphi_image3==-6]=np.nan
    radius,azimuth=polar_coord(im,xcenter,ycenter,angle_resolution)
    r_f,a_f=polar_coord_float(im,xcenter,ycenter,angle_resolution)
    
    os.chdir('/Users/jiliangliu/Dropbox/Aug_18/')
    from peak_finding import oneD_intensity
    I=oneD_intensity(im=im,xcenter=xcenter,ycenter=ycenter,mask=mask).cir_ave()
    I=I[0:r_max]
    
    iso_judge=np.zeros((len(I),))
    iso_local_std=np.zeros((len(I),))
    iso_std=np.zeros((len(I),))
    delta_width = 1
    
    qphi_image1[qphi_image1==1e-6]=np.nan
    
    # standard deviation threshold deteremination
    #calculation of Ave_I bases on qphi_image1 or qphi_image3 is not certain, for very strong diffraction maybe qphi_image3 (log version) is better which could enable judgement on the sigma_loc
    iso_judge1=np.zeros((r_max,))
    for i in range(r_max):
        if size(qphi_image3[isnan(qphi_image3[:,i])==0,i])>angle_resolution/10:
            iso_judge1[i] = np.std(qphi_image1[isnan(qphi_image1[:,i])==0,i])
        else: iso_judge1[i] = np.nan
        
        if i >delta_width and i<len(I)-delta_width:
            Ave_I=np.sum(qphi_image3[:,i-delta_width:i+delta_width],axis=1)/(2*delta_width)
        elif i<delta_width:
            Ave_I=np.sum(qphi_image3[:,i:i+delta_width],axis=1)/delta_width
        elif i>len(I)-delta_width:
            Ave_I=np.sum(qphi_image3[:,i-delta_width:i],axis=1)/delta_width
        #Ave_I[isnan(Ave_I)==0]=sm.nonparametric.lowess(Ave_I,np.arange(0,bins,1),frac=0.06)[:,1]
        Ave_I[Ave_I<nanmean(Ave_I)-.7*np.abs(nanmean(Ave_I))]=np.nan
        Ave_I_mean=np.concatenate((Ave_I[isnan(Ave_I)==0],Ave_I[isnan(Ave_I)==0],Ave_I[isnan(Ave_I)==0]))
        Ave_I_local_std=np.zeros((len(Ave_I[isnan(Ave_I)==0]),))
        for h in range(len(Ave_I[isnan(Ave_I)==0]),2*len(Ave_I[isnan(Ave_I)==0])):
            Ave_I_local_std[h-len(Ave_I[isnan(Ave_I)==0])] = np.std(Ave_I_mean[h-int(angle_resolution)/100:h+int(angle_resolution)/100])
        iso_local_std[i]=np.abs(np.nanmean(Ave_I_local_std))
        iso_std[i]=np.std(Ave_I_mean[len(Ave_I[isnan(Ave_I)==0]):2*len(Ave_I[isnan(Ave_I)==0])]) 
    x,y =histogram(iso_judge1[isnan(iso_judge1)==0],bins=30)
    y=y[:-1]+np.diff(y)/2
    
    from scipy.misc import factorial
    def poisson_func(k,lamb):
        return ((lamb*1.)**k/factorial(k))*np.exp(-lamb)
        
    from scipy.optimize import curve_fit
    k=np.arange(1,31,1)
    
    #return x,y,k,
    if size(x[isnan(x)])>0 or size(x[x==0])==len(x):
        return np.zeros(shape(im))
    lamb=curve_fit(poisson_func,k,x*1./np.sum(x),bounds=[0,inf])[0][0]
    #if lamb<1:
    #    lamb=1
    x1,y1 =histogram((iso_local_std/iso_std)[(np.abs(iso_local_std/iso_std)!=inf)&(isnan(iso_local_std/iso_std)==0)],bins=30)
    y1=y1[:-1]+np.diff(y1)/2
    lamb1=curve_fit(poisson_func,k,x1*1./np.sum(x1),bounds=[np.argmax(x1)-1,np.argmax(x1)+1])[0][0]
        
    def polar_gaussian(r,a,mu_r,mu_a,sigma_r,sigma_a,A,mask):
        return mask*A*exp(-(r-mu_r)**2/sigma_r**2/2)*(exp(-(a-mu_a)**2/sigma_a**2/2)+exp(-(np.round(np.max(a))+a-mu_a)**2/sigma_a**2/2)+exp(-(-np.round(np.max(a))+a-mu_a)**2/sigma_a**2/2))    
    
    def TwoD_gaussian(r,a,mu_r,mu_a,sigma_r,sigma_a,A,roll_index,mask):
        return np.roll(mask*A*exp(-(r-mu_r)**2/sigma_r**2/2)*exp(-(a-mu_a)**2/sigma_a**2/2),int(roll_index),axis=0)  
        
    ###
    if (np.abs(1-y1[int(lamb1)])<bkgd_threshold):
        aniso_bkgd_judge=False
        # always apply two fold sym first
        '''
        im3=np.append(np.append(im,np.fliplr(im),axis=1),np.append(flipud(im),np.fliplr(flipud(im)),axis=1),axis=0)
        imul=np.zeros((shape(im3)))*np.nan
        imul[:shape(im)[0],:shape(im)[1]]=np.copy(im)
        imul = np.roll(np.roll(imul,int(shape(im)[1]-np.round(xcenter)),axis=-1),int(np.round(shape(im)[0]-ycenter)),axis=0)
        
        imdr=np.zeros((shape(im3)))*np.nan
        imdr[:shape(im)[0],:shape(im)[1]]=np.copy(np.fliplr(flipud(im)))
        imdr = np.roll(np.roll(imdr,int(np.round(xcenter)),axis=-1),int(np.round(ycenter)),axis=0)
        shift_index_c=0
        shift_index_r=0
        shift_fitness=inf
        for shift_c in np.arange(-3,4,1):
            for shift_r in np.arange(-3,4,1):
                if shift_fitness>nanmean(np.abs(imul-np.roll(np.roll(imdr,shift_c,axis=-1),shift_r,axis=0))):
                    shift_fitness=nanmean(np.abs(imul-np.roll(np.roll(imdr,shift_c,axis=-1),shift_r,axis=0)))
                    shift_index_c=int(shift_c)
                    shift_index_r=int(shift_r)
        imdr = np.roll(np.roll(imdr,shift_index_c,axis=-1),shift_index_r,axis=0)
        imul[isnan(imul)]=imdr[isnan(imul)]
        imul = np.roll(np.roll(imul,-(int(shape(im)[1]-np.round(xcenter))),axis=-1),-(int(np.round(shape(im)[0]-ycenter))),axis=0)
        im=np.copy(imul[:shape(im)[0],:shape(im)[1]])
        '''
        #sys.exit()
        if (lamb>4):
            # diffraction includes no sharp or intermediate reflections and isotropic background
            for l in range(r_min,r_max):
                IR = im[radius==l]
                IR[isnan(IR)] = I[l]
                im[radius==l] = IR
            return im
            #figure(6),imshow(im)
            #plt.show()
            print 'no reflections'
            #sys.exit()
        elif (lamb<=4):
            diffuse=False
            iso_judge=((iso_judge1>y[np.round(lamb+lamda_threshold_1*lamb**.5).astype(int)])&((iso_local_std/iso_std)<y1[int(lamb1-lamda_threshold_2*lamb1**.5)])).astype(int)
            pass
    elif (np.abs(1-y1[int(lamb1)])>bkgd_threshold):
        # diffraction includes anisotropic background
        if lamb>4:
            aniso_bkgd_judge=True
            diffuse=False
            #im1=transform.rescale(np.copy(im),.1)
            im1=np.copy(im)
            im1[isnan(im1)==1]=0
            U,s,V=linalg.svd(im1)
            L=np.zeros((shape(im1)))
            for i in range(2):
                L=L+np.outer(U[:,i],V[i,:])*(np.diag(s)[np.diag(s)!=0])[i]
            # first three principle components contain low frequency information
            wind_siz=5
            TwoD_local_std_normalized=(im1/L-1)**2+signal.convolve2d((im1/L-1)**2,np.ones((wind_siz,wind_siz)),mode='same')*(im1/L-1)/size(np.ones((wind_siz,wind_siz)))-2*signal.convolve2d((im1/L-1),np.ones((wind_siz,wind_siz)),mode='same')*(im1/L-1)/size(np.ones((wind_siz,wind_siz)))
            # if noise increase as signal increasing, then normalize local std by dividing by low frequency component offsetting noise enhancement due to low frequency domain.
            TwoD_local_std=(im1-L)**2+signal.convolve2d((im1-L)**2,np.ones((wind_siz,wind_siz)),mode='same')*(im1-L)/size(np.ones((wind_siz,wind_siz)))-2*signal.convolve2d((im1-L),np.ones((wind_siz,wind_siz)),mode='same')*(im1-L)/size(np.ones((wind_siz,wind_siz)))
            x2,y2 =histogram(log(TwoD_local_std*TwoD_local_std_normalized)[(isnan(log(TwoD_local_std*TwoD_local_std_normalized))==0)],bins=30)
            y2=y2[:-1]+np.diff(y2)/2
            k=np.arange(1,31,1)
            lamb2=curve_fit(poisson_func,k,x2*1./np.sum(x2),bounds=[1,inf])[0][0]
            #L1 = np.copy(im1) # the residue after background subtraction may not leads to a gaussian distribution if there is no sharp reflections, all low frequence componen include noise had been reduced
            L1 = np.copy(L)
            L2 = np.copy(im1)
            L1_mask=((log(TwoD_local_std*TwoD_local_std_normalized))>y2[np.round(lamb2+1*lamb2**.5).astype(int)])
            L2_mask=((log(TwoD_local_std*TwoD_local_std_normalized))>y2[np.round(lamb2+1*lamb2**.5).astype(int)])
            L2_mask=ndimage.binary_dilation(L2_mask,np.ones((20,20)))
            L1[L1_mask]=np.nan
            L2[L2_mask]=np.nan
            
            #down_sample=.1
            L1=transform.rescale(L1,down_sample)
            qphi_image_L=qphi_image(L1,xcenter*down_sample,ycenter*down_sample,int(r_min*down_sample),int(r_max*down_sample),angle_resolution)
            qphi_image_L=qphi_image_L[:,:int(r_max*down_sample)]
            qphi_image_L[qphi_image_L==0]=np.nan
            #qphi_image_L2=qphi_image(L2,xcenter,ycenter,r_min,r_max,angle_resolution)
            #qphi_image_L2=transform.resize(qphi_image_L2,shape(qphi_image_L))
            #qphi_image_L2=qphi_image_L[:,:2000]
            qphi_image_L2=np.copy(qphi_image_L)
            # background sym determine
            bkgd_sym_ssd=inf
            bkgd_sym_opt=np.array([2,4,6,8])
            bkgd_sym=2
            #bkgd_fuck=np.zeros((4,))
            #bkdg_fit_bias=4
            for num1 in bkgd_sym_opt:
                I6=np.zeros((angle_resolution/num1,int(r_max*down_sample),num1))
                for k in range(num1):
                    I6[:,:,k] = np.copy(qphi_image_L[k*angle_resolution/num1:(k+1)*angle_resolution/num1,:])
                I7 = np.copy(I6)
                I7[I7==0]=np.nan
                TD_model_L=nanmean(I7,axis=2)
                for kk in range(num1):
                    if kk==0:
                        qphi_image_L_model=TD_model_L
                    else:
                        qphi_image_L_model=np.append(qphi_image_L_model,TD_model_L,axis=0)
                kk=(isnan(I7)==0).astype(int)
                kk=np.sum(kk,axis=2)
                #bkgd_fuck[num1/2-1]=np.nanmean(np.abs(qphi_image_L_model-qphi_image_L)[(qphi_image_L_model-qphi_image_L)!=0])*(2-1.*size(kk[kk>0])/size(kk))**bkdg_fit_bias
                if bkgd_sym_ssd>np.nanmean(np.abs(qphi_image_L_model-qphi_image_L)[(qphi_image_L_model-qphi_image_L)!=0])*(2-1.*size(kk[kk>0])/size(kk))**bkdg_fit_bias:
                    bkgd_sym_ssd=np.nanmean(np.abs(qphi_image_L_model-qphi_image_L)[(qphi_image_L_model-qphi_image_L)!=0])*(2-1.*size(kk[kk>0])/size(kk))**bkdg_fit_bias
                    bkgd_sym=num1
                else:
                    bkgd_sym_ssd=bkgd_sym_ssd
                    bkgd_sym=bkgd_sym
            qphi_image_L[isnan(qphi_image_L)]=np.roll(qphi_image_L,angle_resolution/2,axis=0)[isnan(qphi_image_L)]
            IB1=np.zeros((shape(qphi_image_L)[0]/bkgd_sym,shape(qphi_image_L)[1],bkgd_sym))
            for k in range(bkgd_sym):
                IB1[:,:,k] = np.copy(qphi_image_L[k*angle_resolution/bkgd_sym:(k+1)*angle_resolution/bkgd_sym,:])
            IB1[IB1==-6]=np.nan
            TwoD_B_model=nanmean(IB1,axis=2)
            #print time.time()-t;plt.show();sys.exit()
        
        
            TwoD_B_model[isnan(TwoD_B_model)]=0  
            TwoD_B_healing=np.copy(TwoD_B_model).astype(uint16)
            #from skimage.filters import rank
            #for i in range(30):
            while size(TwoD_B_healing[TwoD_B_healing==0])>0:
                TwoD_B_healing = rank.mean(TwoD_B_healing.astype(uint16),np.ones((5,5)),mask=(TwoD_B_healing!=0))
            TwoD_B_healing=TwoD_B_healing.astype(float)
            TwoD_B_model[TwoD_B_model==0]=TwoD_B_healing[TwoD_B_model==0]
            for k in range(bkgd_sym):
                IB1[:,:,k][isnan(IB1[:,:,k])==1]=TwoD_B_model[isnan(IB1[:,:,k])==1]
                qphi_image_L2[k*angle_resolution/bkgd_sym:(k+1)*angle_resolution/bkgd_sym,:]=IB1[:,:,k]
            
            #from skimage import transform
            qphi_image_L2=transform.resize(qphi_image_L2,shape(qphi_image1))
            qphi_mask3=qphi_image(L2_mask,xcenter,ycenter,r_min,r_max,angle_resolution)
            qphi_mask3[qphi_mask3<1]=0
            qphi_image1[qphi_mask3!=1]=np.nan
            qphi_image1=qphi_image1-qphi_image_L2
            qphi_image1[(isnan(qphi_image1))]=1
            qphi_image1[qphi_image1==0]=1
            qphi_image1[(qphi_image1<=1e-6)]=1
            qphi_image1[isnan(qphi_mask[:,:r_max])]=np.nan
            qphi_image1[qphi_mask2[:,:r_max]]=1e-6
            #print time.time()-t
            #np.roll could apply on 2D image, 
            #but skimage inpaint is not very well perfomred on x-ray data reconstruction
            #qphi_image1[qphi_image1==0]=1e-6
            qphi_image2=log(np.roll(qphi_image1,90,axis=0))
            qphi_image2[qphi_image2<-6]=-6
            mask2=(qphi_image2==-6)
            mask2=ndimage.binary_dilation(mask2,np.ones((2,2)))
            qphi_image2[mask2]=-6
            qphi_image3=np.copy(qphi_image2)
            qphi_image3[qphi_image3==-6]=np.nan
            
            iso_judge=np.zeros((len(I),))
            IB_judge1=np.zeros((len(I),))
            for i in range(r_max):
                IB_judge1[i]=np.std(qphi_image1[qphi_image1[:,i]>1,i])
            x3,y3=histogram(IB_judge1[isnan(IB_judge1)==0],bins=30)
            lamb3=curve_fit(poisson_func,np.arange(1,31,1),x3*1./np.sum(x3),bounds=[1,inf])[0][0]
            if lamb3>4:
                pass
            elif lamb3<=4:
                IB_judge=oneD_intensity(im=L2_mask.astype(int),xcenter=xcenter,ycenter=ycenter,mask=mask).cir_ave()
                iso_judge=(IB_judge[:r_max]>(nanmean(IB_judge[:r_max][IB_judge[:r_max]>0])+1.*np.std(IB_judge[:r_max][IB_judge[:r_max]>0])))
        elif lamb<=4:# here anisotropic is caused by the background noise no due to symmetrical fold 
            aniso_bkgd_judge=False
            diffuse=True
            iso_judge=((iso_judge1>y[np.round(lamb+lamda_threshold_3*lamb**.5).astype(int)])&((iso_local_std/iso_std)<.8)).astype(int)
            print 'diffuse diffraction'
    
    def consecutive(data,stepsize=1):
        return np.split(data.T,np.where(np.diff(data) != stepsize)[0]+1)
    aniso_span = consecutive(np.where(iso_judge)[0])
    if size(aniso_span)==0:
        if aniso_bkgd_judge==False:
            for l in range(r_min,r_max):
                IR = im[radius==l]
                IR[isnan(IR)] = I[l]
                im[radius==l] = IR
            figure(6),imshow(im)
            plt.show()
            print 'iso bkgd no symmetrical folds sharp'
        elif aniso_bkgd_judge==True:
            qphi_image6=np.copy(qphi_image_L2)
            for l in range(r_min,r_max):
                IR = im[radius==l]
                FR = interp(a_f[radius==l],np.arange(0,angle_resolution,1)[isnan(qphi_image6[:,l])==0],qphi_image6[isnan(qphi_image6[:,l])==0,l])
                IR[isnan(IR)]=FR[isnan(IR)]
                im[radius==l]= IR 
            #figure(6),imshow(im)
            #plt.show()
            print 'aniso bkgd no symmetrical folds sharp'
            print 'aniso_bkgd symmetric'
            print bkgd_sym 
            return im
        #sys.exit()
    
    
    
    
    #sym=np.zeros((len(aniso_span),))
    #radius,azimuth=polar_coord(im,xcenter,ycenter,bins)
    for i in range(len(aniso_span)):
        
        if aniso_bkgd_judge==True:
            #because here already subtract the background and enhance the signal, hence namean is enough
            I1=np.nanmean(qphi_image3[:,aniso_span[i]],axis=1)
        elif aniso_bkgd_judge==False:
            I1=np.sum(qphi_image3[:,aniso_span[i]],axis=1)/len(aniso_span[i])
            if size(I1[isnan(I1)==0])<angle_resolution/8:
                II = (isnan(qphi_image3[:,aniso_span[i]])).astype(int)
                II = np.sum(II,axis=1)
                II = (II>=(len(aniso_span))/2)
                I1=nanmean(qphi_image3[:,aniso_span[i]],axis=1)
                if size(II)>(angle_resolution-angle_resolution/16):
                    pass
                else: I1[~II]=np.nan
            if np.std(I1[isnan(I1)==0])>np.std(nanmean(qphi_image3[:,aniso_span[i]],axis=1)[isnan(nanmean(qphi_image3[:,aniso_span[i]],axis=1))==0]):
                II = (isnan(qphi_image3[:,aniso_span[i]])).astype(int)
                II = np.sum(II,axis=1)
                II = (II>=(len(aniso_span))/2)
                I1=nanmean(qphi_image3[:,aniso_span[i]],axis=1)
                I1[~II]=np.nan
                I1=nanmean(qphi_image3[:,aniso_span[i]],axis=1)
    
        I2=np.copy(I1)
        I2=I2-nanmean(I1)
        I2[I2<0]=0
        I2[isnan(I2)]=0
        I3=np.correlate(I2,I2,mode='same')
        sym_fold=np.argsort(np.fft.rfft(I3)[2:10])[::-1]+2
        sym_fold=sym_fold[sym_fold%2==0]# here I enforce the symmtry with even fold
        Va=np.zeros((4,))
        sym=0
        fold_bias=fold_bias
        
        if aniso_bkgd_judge==False:
            if diffuse==False:
                fittness_threshold=fittness_threshold# for **.5 use 16, **1 use 32
            elif diffuse==True:
                fittness_threshold=fittness_threshold/2.
                fold_bias=fold_bias/2.
        else:fittness_threshold=fittness_threshold/4.
        
        for j in range(len(sym_fold)):
            # inequal background and tiltation of sample caused aniso distribution require the normalization here
            model=nanmean(reshape(I1,(sym_fold[j],angle_resolution/sym_fold[j])).T,axis=1)
            K=reshape(I1,(sym_fold[j],angle_resolution/sym_fold[j])).T
            kk=isnan(K)==0
            kk=np.sum(kk.astype(int),axis=1)
            Va[j] = nanmean(np.abs(I1-np.tile(model,sym_fold[j]))**2)**1*(2-1.*sym_fold[j]*size(kk[kk>1])/angle_resolution)**fold_bias
        
        Va[Va==0]=6
        fittness_porior=True
        if mean(iso_judge1[isnan(iso_judge1)==0])/mean(I[isnan(I)==0])/fittness_threshold>extreme_fitness:
            extreme_fitness=extreme_fitness
        else: extreme_fitness=mean(iso_judge1[isnan(iso_judge1)==0])/mean(I[isnan(I)==0])/fittness_threshold
        
        if size(Va[Va<extreme_fitness])>1:
            sym = np.max(sym_fold[np.where(Va<extreme_fitness)[0]])
        else:    
            if fittness_porior==True:
                sym = sym_fold[np.argmin(Va)]
            else: # try to bias to  maximum symmtric   
                if np.min(np.round(np.log(Va)))<=np.round(log(mean(iso_judge1[isnan(iso_judge1)==0])/mean(I[isnan(I)==0])/fittness_threshold)):#-3.:
                    if size(np.unique(np.log(Va)))>1:
                        sym = np.max(sym_fold[np.where(np.round(log(Va))==np.min(np.unique(np.round(log(Va)))))[0]])
                        if np.max(sym_fold[np.where(np.floor(log(Va))==np.min(np.unique(np.floor(log(Va)))))[0]])>sym:
                            sym=np.max(sym_fold[np.where(np.floor(log(Va))==np.min(np.unique(np.floor(log(Va)))))[0]])
                    else: sym = np.max(sym_fold[:4])
                else: sym=0
        
        if (size(Va[Va==6])>0) & (size(Va[Va<extreme_fitness])<1):
            if (size(Va[Va==6])==1) and (np.min(np.round(np.log(Va)))>np.round(log(mean(iso_judge1[isnan(iso_judge1)==0])/mean(I[isnan(I)==0])/fittness_threshold))):
                sym = sym_fold[Va==6][0]
            elif size(Va[Va==6])>1:
                sym=np.max(sym_fold[np.where(Va==6)[0]]) 
        
        if sym>0:
            I6=np.zeros((angle_resolution/sym,len(aniso_span[i]),sym))
            for k in range(sym):
                I6[:,:,k] = np.copy(qphi_image2[k*angle_resolution/sym:(k+1)*angle_resolution/sym,aniso_span[i]])
            print sym,' ',i
            I7 = np.copy(I6)
            I7[I7==-6]=np.nan
            nonnan_size=np.zeros((sym,))
            for n in range(sym):
                nonnan_size[n] = size(I7[isnan(I7[:,:,n])==0,n])
            TwoD_model = I7[:,:,np.argmax(nonnan_size)]
            sym_region_shift_r=(np.zeros((sym,))).astype(int)
            sym_region_shift_c=(np.zeros((sym,))).astype(int)
            if fitting_shift == True:
                for n in range(sym):
                    sym_region_fitness=inf
                    if n!=np.argmax(nonnan_size):
                        for m in np.arange(-5,5,1):
                            for mm in np.arange(-5,5,1):
                                if sym_region_fitness>nanmean(np.abs(TwoD_model-(np.roll(np.roll(qphi_image3,m,axis=-1),mm,axis=0))[n*angle_resolution/sym:(n+1)*angle_resolution/sym,aniso_span[i]])):
                                    sym_region_fitness=nanmean(np.abs(TwoD_model-(np.roll(np.roll(qphi_image3,m,axis=-1),mm,axis=0))[n*angle_resolution/sym:(n+1)*angle_resolution/sym,aniso_span[i]]))
                                    sym_region_shift_c[n]=m
                                    sym_region_shift_r[n]=mm
                        I7[:,:,n]=(np.roll(np.roll(qphi_image3,sym_region_shift_c[n],axis=-1),sym_region_shift_r[n],axis=0))[n*angle_resolution/sym:(n+1)*angle_resolution/sym,aniso_span[i]]
                
            if np.sum(np.sum(nanmean(I7,axis=2)))==NaN:
                if diffuse==False:
                    from skimage import restoration
                    TwoD_model = restoration.inpaint_biharmonic(nanmean(I7,axis=2),mask=(isnan(nanmean(I7,axis=2))))
                elif diffuse==True:
                    TwoD_model=nanmean(I7,axis=2)
                    TwoD_healing=exp(np.copy(TwoD_model))
                    TwoD_healing[isnan(TwoD_healing)]=0
                    TwoD_model[isnan(TwoD_model)]=0
                    while size(TwoD_healing[TwoD_healing==0])>0:
                        TwoD_healing=rank.mean(TwoD_healing.astype(uint16),np.ones((5,5)),mask=(TwoD_healing!=0))
                    TwoD_model[TwoD_model==0]=log((TwoD_healing[TwoD_model==0]).astype(float))
            
            I6[I6==-6]=np.nan
            for k in range(sym):
                fit_model=np.zeros((shape(TwoD_model)))
                fit_model=np.roll(np.roll(TwoD_model,-sym_region_shift_c[k],axis=-1),-sym_region_shift_r[k],axis=0)
                fit_model[:,:sym_region_shift_c[k]]=np.copy(TwoD_model[:,:sym_region_shift_c[k]])
                I6[:,:,k][isnan(I6[:,:,k])==1]=fit_model[isnan(I6[:,:,k])==1]
                qphi_image2[k*angle_resolution/sym:(k+1)*angle_resolution/sym,aniso_span[i]]=I6[:,:,k]
            #iso_judge[aniso_span[i]]=1*sym   
        else:# here anisotropic is caused by the background noise no due to symmetrical fold 
            iso_judge[aniso_span[i]]=0
    
    if aniso_bkgd_judge==True:
        qphi_image2[isnan(qphi_image2)]=0
        qphi_image6=exp(np.roll(qphi_image2,-90,axis=0))+qphi_image_L2
    else: qphi_image6=exp(np.roll(qphi_image2,-90,axis=0))
    
    
    
    
    #print time.time()-t#3
    #radius,azimuth=polar_coord(im,xcenter,ycenter,bins)
    aniso_span1 = consecutive(np.where(iso_judge)[0])
    aniso_area=aniso_span1[0]
    for s in range(1,len(aniso_span1)):
        aniso_area=np.append(aniso_area,aniso_span1[s])
    
    
    im2=np.copy(im)
    if aniso_bkgd_judge==True:
        for l in range(r_min,r_max):
            IR = im2[radius==l]
            FR = interp(a_f[radius==l],np.arange(0,angle_resolution,1)[isnan(qphi_image6[:,l])==0],qphi_image6[isnan(qphi_image6[:,l])==0,l])
            IR[isnan(IR)]=FR[isnan(IR)]
            im2[radius==l]= IR  
        print 'aniso_bkgd symmetric'
        print bkgd_sym 
    elif aniso_bkgd_judge==False:
        for l in range(r_min,r_max):
            if l in aniso_area:
                IR = im2[radius==l]
                FR = interp(a_f[radius==l],np.arange(0,angle_resolution,1)[isnan(qphi_image6[:,l])==0],qphi_image6[isnan(qphi_image6[:,l])==0,l])
                IR[isnan(IR)]=FR[isnan(IR)]
                im2[radius==l]= IR
            else: 
                IR = im[radius==l]
                IR[isnan(IR)] = I[l]
                im2[radius==l] = IR
    #print time.time()-t
    return im2#,I,iso_judge#,r_f,lamb,lamb1,iso_local_std,iso_std,y,y1,x,x1,iso_judge1


#im3,I,IB_judge1,iso_jugde,lamb2 = image_healing(im,xcenter,ycenter,mask=mask,angle_resolution=720,r_max=2000,r_min=20,bkdg_fit_bias=4,fold_bias=4,fittness_threshold=64,extreme_fitness=1e-3,down_sample=.1)
#im3 = image_healing(im,xcenter,ycenter,mask=mask,angle_resolution=720,r_max=2000,r_min=20,bkdg_fit_bias=4,fold_bias=4,fittness_threshold=64,extreme_fitness=1e-3,down_sample=.1)
