import sys
import numpy as np


class oneD_intensity:
    def __init__(self,**kwargs):
        if kwargs['im'] is None:
            print ('not image input')
            sys,exit()
        else: self.im=kwargs['im']
        if kwargs['xcenter'] is None:
            print ('center of diffraction is unkonwn')
            sys.exit()
        else: self.xcenter=float(kwargs['xcenter'])
        if kwargs['ycenter'] is None:
            print ('center of diffraction is unkonwn')
            sys.exit()
        else: self.ycenter=float(kwargs['ycenter'])
        #set maximum intensity threshold
        im = self.im
        self.in_max = kwargs['in_max'] if 'in_max' in kwargs else np.inf
        #set minimum intensity threshold
        self.in_min = kwargs['in_min'] if 'in_min' in kwargs else -np.inf
        #set custom mask
        self.mask = kwargs['mask'] if 'mask' in kwargs else np.zeros(self.im.shape)
        #set maximum wide angle for caculation
        self.radius_bins = kwargs['bins'] if 'bins' in kwargs else max(self.im.shape)
        self.radius = np.zeros(im.shape)
        self.azimuth = np.zeros(im.shape)
        self.mask = kwargs['mask'] if 'mask' in kwargs else None
        self.dilution = True if 'dilution' in kwargs else False
        self.dilution_size = kwargs['dilution_size'] if 'dilution_size' in kwargs else None


    def cir_ave(self):

        def polar_coord(self):
            shape_ind = np.asarray(np.shape(self.im))
            x = np.arange(1,shape_ind[1]+1,1)
            y = np.arange(1,shape_ind[0]+1,1)
            #xx is column index of input image, yy is row index of input image
            xx , yy = np.meshgrid(x, y)
            # produce the x-coord for matrix with origin at scattering center
            xx = xx - self.xcenter
            # produce the y-coord for matrix with origin at scattering center
            yy = yy - self.ycenter
            #produce azimuth coord
            azimuth = np.arctan(yy/xx)
            azimuth = azimuth  + (xx<0)*np.pi +((xx>=0)&(yy<0))*2*np.pi
            #produce radius coord
            radius = np.sqrt(xx**2+yy**2)
            radius = np.round(radius)
            radius = radius.astype(int)
            return radius,azimuth

        def make_mask(self):
            if self.mask is None:
                mask1 = self.im > self.in_max
                mask2 = self.im < self.in_min
                if self.dilution == True:
                    dilution_size = int(self.dilution_size)
                    mask1 = ndimage.binary_dilation(mask1,
                               structure=np.ones((dilution_size,dilution_size)))
                    mask2 = ndimage.binary_dilation(mask2,
                               structure=np.ones((dilution_size,dilution_size)))
                mask3 = radius>self.radius_bins
                mask = (mask1)|(mask2)|(mask3)
            else:
                mask1 = self.im > self.in_max
                mask2 = self.im < self.in_min
                if self.dilution == True:
                    dilution_size = int(self.dilution_size)
                    mask1 = ndimage.binary_dilation(mask1,
                               structure=np.ones((dilution_size,dilution_size)))
                    mask2 = ndimage.binary_dilation(mask2,
                               structure=np.ones((dilution_size,dilution_size)))
                mask3 = radius>self.radius_bins
                mask = (mask1)|(mask2)|(mask3)
                mask = (self.mask)|(mask1)|(mask2)|(mask3)
            return mask

        radius,azimuth = polar_coord(self)
        im = self.im
        mask = make_mask(self)
        mask = mask>0
        a = radius[~mask]
        b = im[~mask]
        val = np.zeros((self.radius_bins+1,))
        #for j in range(self.radius_bins):
        #    val[j] = accum_np(a, b, j)

        intensity = np.bincount(a.flatten(),weights=b.flatten())
        counts = np.bincount(a.flatten())

        val[0:len(intensity)] = intensity/counts

        return val
