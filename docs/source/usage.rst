=====
Usage
=====

Start by importing healing_scattering_image.

.. code-block:: python

    import healing_scattering_image

Functions
--------------------------
.. autofunction:: healing_scattering_image.heal_scatter_image.heal_scatter_image

Example
--------------------------
.. ipython:: python

   import numpy as np
   from skimage.draw import polygon
   import matplotlib.pyplot as plt
   from scipy.io import loadmat
   from scipy import ndimage
   from skimage.filters import median
   from skimage.transform import resize
   import healing_scattering_image
   from healing_scattering_image.heal_scatter_image import heal_scatter_image
   import os
   from pkg_resources import resource_filename

   path = resource_filename('healing_scattering_image',\
                            'example_data/example_1.npz')
   sample_name = path
   im = np.double(np.load(sample_name)['im'])
   im += 1.
   mask = np.load(sample_name)['mask']

   from skimage.transform import resize

   #im = resize(im,(int(np.shape(im)[0]/2),int(np.shape(im)[1]/2)))
   #mask = resize(mask,(int(np.shape(mask)[0]/2),int(np.shape(mask)[1]/2)))

   mask = mask.astype(bool)

   im[mask] = np.nan

   xcenter = (np.load(sample_name)['xcenter'])#/2
   ycenter = (np.load(sample_name)['ycenter'])#/2
   r_min = 10
   r_max = np.max(im.shape)#int(np.max(np.shape(im)))
   angle_resolution = 360
   delta_width = 2
   bkgd_threshold  = 0.3
   peaks_threshold = 4
   healed_im, aniso_place, sym_record,\
   iso_judge_global , iso_local_vs_global,\
   qphi_image_6, I = heal_scatter_image(im,mask,
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
                                         fold_bias = 6.,
                                         fittness_threshold = 1.,
                                         extreme_fitness = 1e-3,
                                         two_fold_apply = True,
                                         fittness_prior=True,
                                         down_sample = 0.1,)


Plots
--------------------------
.. plot::

   import numpy as np
   from skimage.draw import polygon
   import matplotlib.pyplot as plt
   from scipy.io import loadmat
   from scipy import ndimage
   from skimage.filters import median
   from skimage.transform import resize
   import healing_scattering_image
   from healing_scattering_image.heal_scatter_image import heal_scatter_image
   import os
   from pkg_resources import resource_filename

   path = resource_filename('healing_scattering_image',\
                            'example_data/example_1.npz')
   sample_name = path
   im = np.double(np.load(sample_name)['im'])
   im += 1.
   mask = np.load(sample_name)['mask']

   from skimage.transform import resize

   #im = resize(im,(int(np.shape(im)[0]/2),int(np.shape(im)[1]/2)))
   #mask = resize(mask,(int(np.shape(mask)[0]/2),int(np.shape(mask)[1]/2)))

   mask = mask.astype(bool)

   im[mask] = np.nan

   xcenter = (np.load(sample_name)['xcenter'])#/2
   ycenter = (np.load(sample_name)['ycenter'])#/2
   r_min = 10
   r_max = np.max(im.shape)#int(np.max(np.shape(im)))
   angle_resolution = 360
   delta_width = 2
   bkgd_threshold  = 0.3
   peaks_threshold = 4
   healed_im, aniso_place, sym_record,\
   iso_judge_global , iso_local_vs_global,\
   qphi_image_6, I = heal_scatter_image(im,mask,
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
                                         fold_bias = 6.,
                                         fittness_threshold = 1.,
                                         extreme_fitness = 1e-3,
                                         two_fold_apply = True,
                                         fittness_prior=True,
                                         down_sample = 0.1,)

   fig,ax = plt.subplots(1,2)
   ax[0].imshow(np.log(im))
   ax[1].imshow(np.log(healed_im))
   plt.show()
