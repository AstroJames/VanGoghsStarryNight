''' This file takes the power spectrum of the Starry night
Van Gogh image, and plots the azimuthal and 2D spectrum'''

import os                                  # command line
import argparse                            # command line arguments
import numpy as np                         # regular mathematics stuff
import pandas as pd                        # data handling
import matplotlib.pyplot as plt            # visualisation
from matplotlib import rc                  # nicer text in matplotlib
import imageio                             # reading in image data
import h5py                                # importing in hdf5 files
import skimage                             # import image data
from skimage import measure, filters       # for drawing contours and Gaussian filters
from scipy import fftpack, misc            # fourier transform
from sklearn import linear_model           # linear regression for measuring scaling
from sklearn.metrics import mean_squared_error, r2_score

# Command line arguments
############################################################################################################################################
ap 			= argparse.ArgumentParser(description = 'Just a bunch of input arguments')
ap.add_argument('-file','--file',required=True,help='the file name for the input file',type=str)
args 		= vars(ap.parse_args())
############################################################################################################################################


def azimuthalAverage(image, center=None, variance=None):
    """
    Calculate the azimuthally averaged radial profile.

    Modified from: http://www.astrobetter.com/wiki/python_radial_profiles

    image       - The 2D image
    center      - The [x,y] pixel coordinates used as the center. The default is
                None, which then uses the center of the image (including
                fractional pixels).
    variance    - true / false if the variance of the averaging should also
                be calculated.
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    # pick the median of the image data as the center
    if not center:

        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    # calculate the sqrt( (x-center)^2 + (y-center)^2 )
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii

    ind         = np.argsort(r.flat) # sort the flattened array and store indices
    r_sorted    = r.flat[ind]        # store the sorted array
    i_sorted    = image.flat[ind]    # store the sorted pixel intensities
    i_float     = i_sorted.astype(float) # store the intensities as floats
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar  = r_int[1:] - r_int[:-1]    # assumes all different radii represented
    rind    = np.where(deltar)[0]       # location of when the bins change
    nr      = rind[1:] - rind[:-1]      # number of each radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_float) # creates a cumulative sum vector
    tbin = csim[rind[1:]] - csim[rind[:-1]] # finds the difference between each of the radii bins

    # The expected value of the azimuthal average
    Exp_val = tbin / nr

    # Calculate the variance
    Var_val     = np.zeros(len(Exp_val))
    Var_index   = 0

    # This is the slow way to calculate the variance, but oh well.
    # FIX
    if variance:
        for bin in np.unique(r_int[1:]):

            try:
                Var_val[Var_index] = np.var(i_float[r_int==bin])
                Var_index += 1
            except:
                print("The variance calculation has finished.")

        return Exp_val, Var_val


    return Exp_val

def FourierTransform(data,n,type,variance=None):
    """
    Calculate the fourier transform.

    data        - The 2D image data
    n           - the coef. of the fourier transform = grid size.
    type        - either '2D' or 'aziAverage' for 2D or averaged power spectrum calculations.
    variance    - passed to the azimuthal averaging if you want to calculate the variance of
                the average.
    """
    data    = data.astype(float)                    # make sure the data is float type
    data    = ( 1/(n)**2 ) * fftpack.fft2(data)     # 2D fourier transform
    data    = fftpack.fftshift(data)                # center the transform so k = (0,0) is in the center
    data    = np.abs(data*np.conjugate(data))       # take the power spectrum

    # take the azimuthal average of the powr spectrum
    if type == 'aziAverage':

        if not variance:

            data  = azimuthalAverage(data)

            return data
        else:

            data, var  = azimuthalAverage(data,variance=True)

            return data, var

    return data

def linear_regression(X,Y,x):
    """
    Calculate the slope of some data that needs to be log transformed.

    X        - numpy array of the selected X variable
    Y        - numpy array of the selected Y variable
    Xreal    - the X domain that you want to plot it over (just in case you want a Data
            set and not just parameters for visualisation purpose). Pretty hacky.
    """

    # make sure data is in the correct structure in and log-log.
    Y      = np.transpose([np.log10(Y)])
    X      = np.transpose([np.log10(X)])

    # Perform the linear regression
    regr            = linear_model.LinearRegression()
    regr.fit(X,Y)

    # Just make a bunch of values over the entire x domain for the line
    slope           = regr.coef_[0][0]
    intercept       = regr.intercept_[0]
    y   = [slope * i + intercept for i in x]

    return slope, intercept, x, y


# Start Script
############################################################################################################################################

image = imageio.imread(args['file'])
n = float(image.shape[0])

# Choose just the blue and green channels to calculate the power spectra.
image_Blue  = np.array(image[:,:,0])
image_Green = np.array(image[:,:,1])
image_Red   = np.array(image[:,:,2])

# Take the 2D power spectra of each of the channels
Blue2DSpectra   = FourierTransform(image_Blue,n,'2D')
Green2DSpectra  = FourierTransform(image_Green,n,'2D')
Red2DSpectra    = FourierTransform(image_Red,n,'2D')

# Take the azimuthally averaged power spectrum, and the variance.
Blue2DSpectra_azi, Blue_var    = FourierTransform(image_Blue,n,'aziAverage',variance=True)
Green2DSpectra_azi, Green_var  = FourierTransform(image_Green,n,'aziAverage',variance=True)
Red2DSpectra_azi, Red_var      = FourierTransform(image_Red,n,'aziAverage',variance=True)

k = np.array(range(0,len(Blue2DSpectra_azi)))

# Figure 2: Starry Night Data
############################################################################################################################################
f, ax = plt.subplots(1, 3, figsize=(6,4), dpi=200, facecolor='white')
f.subplots_adjust(wspace=0.01,hspace=0.01)
fs = 14

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Blue channel
im1 = ax[0].imshow( image_Blue, cmap=plt.cm.plasma )
ax[0].annotate('Blue Channel',xy=(0.02, 0.9),xycoords='axes fraction',
               xytext=(0.02, 0.9),textcoords='axes fraction',
               fontsize=fs,color='w')
ax[0].set_xticks([])
ax[0].set_yticks([])

# Green channel
im2 = ax[1].imshow( image_Green, cmap=plt.cm.plasma )
ax[1].annotate('Green Channel',xy=(0.02, 0.9),xycoords='axes fraction',
               xytext=(0.02, 0.9),textcoords='axes fraction',
               fontsize=fs,color='w')
ax[1].set_xticks([])
ax[1].set_yticks([])

# Red channel
im3 = ax[2].imshow( image_Red, cmap=plt.cm.plasma )
ax[2].annotate('Red Channel',xy=(0.02, 0.9),xycoords='axes fraction',
               xytext=(0.02, 0.9),textcoords='axes fraction',
               fontsize=fs,color='w')
ax[2].set_xticks([])
ax[2].set_yticks([])

# plt.savefig('Figure2.png',dpi=400)
plt.close()

# Figure 3: Power Spectrum and contours
############################################################################################################################################
f, ax = plt.subplots(2, 3, figsize=(6,3), dpi=300, facecolor='white')
f.subplots_adjust(wspace=0.01,hspace=0.01)
fs = 8

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Green channel power spectra
im1 = ax[0,0].imshow( np.log10( Green2DSpectra ), cmap=plt.cm.plasma )
ax[0,0].annotate(r'Green Channel',
                 xy=(0.02, 0.9),xycoords='axes fraction',
                 xytext=(0.02, 0.9),textcoords='axes fraction',
                 fontsize=fs+3,color='w')
ax[0,0].annotate(r'$\log_{10} \mathcal{P}(k_x,k_y):$ ' + '[{},{}]'.format( round( np.log10( Green2DSpectra.min() ),1 ),
                                                                          round( np.log10( Green2DSpectra.max() ),1 ) ),
                 xy=(0.02, 0.05),xycoords='axes fraction',
                 xytext=(0.02, 0.05),textcoords='axes fraction',
                 fontsize=fs,color='w')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])

# Blue channel power spectra
im2 = ax[0,1].imshow( np.log10( Blue2DSpectra ), cmap=plt.cm.plasma )
ax[0,1].annotate(r'Blue Channel',
                 xy=(0.02, 0.9),xycoords='axes fraction',
                 xytext=(0.02, 0.9),textcoords='axes fraction',
                 fontsize=fs+3,color='w')
ax[0,1].annotate(r'$\log_{10} \mathcal{P}(k_x,k_y):$ ' + '[{},{}]'.format( round( np.log10( Blue2DSpectra.min() ),1 ),
                                                                          round( np.log10( Blue2DSpectra.max() ),1) ),
                 xy=(0.02, 0.05),xycoords='axes fraction',
                 xytext=(0.02, 0.05),textcoords='axes fraction',
                 fontsize=fs,color='w')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

# Red channel power spectra
im3 = ax[0,2].imshow( np.log10( Red2DSpectra ), cmap=plt.cm.plasma )
ax[0,2].annotate(r'Red Channel',
                 xy=(0.02, 0.9),xycoords='axes fraction',
                 xytext=(0.02, 0.9),textcoords='axes fraction',
                 fontsize=fs+3,color='w')
ax[0,2].annotate(r'$\log_{10} \mathcal{P}(k_x,k_y):$ ' + '[{},{}]'.format( round( np.log10( Red2DSpectra.min() ),1 ),
                                                                          round( np.log10( Red2DSpectra.max() ),1) ),
                 xy=(0.02, 0.05),xycoords='axes fraction',
                 xytext=(0.02, 0.05),textcoords='axes fraction',
                 fontsize=fs,color='w')
ax[0,2].set_xticks([])
ax[0,2].set_yticks([])


# Now create the smoothed power spectrum to find the isobars

# Standard deviation of the Gaussian kernel
std_dev = 10

im4 = ax[1,0].imshow( filters.gaussian(np.log10( Green2DSpectra ), std_dev), cmap=plt.cm.plasma )
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])

im5 = ax[1,1].imshow( filters.gaussian(np.log10( Blue2DSpectra ), std_dev), cmap=plt.cm.plasma )
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

im6 = ax[1,2].imshow( filters.gaussian(np.log10( Red2DSpectra ), std_dev), cmap=plt.cm.plasma )
ax[1,2].set_xticks([])
ax[1,2].set_yticks([])

# for a bunch of different contours
for parms in np.linspace(-4,0,10): #the contour domain, i.e. the power of each isobar
    contours1 = measure.find_contours( filters.gaussian(np.log10( Green2DSpectra ), 10), parms)
    contours2 = measure.find_contours( filters.gaussian(np.log10( Blue2DSpectra ), 10), parms)
    contours3 = measure.find_contours( filters.gaussian(np.log10( Red2DSpectra ), 10), parms)

    for n, contour in enumerate(contours1):
        ax[1,0].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

    for n, contour in enumerate(contours2):
        ax[1,1].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

    for n, contour in enumerate(contours3):
        ax[1,2].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

#plt.savefig('Figure3.png',dpi=300)
plt.close()

# Figure 4: Azimuthally-averaged power spectrum
############################################################################################################################################
f, ax = plt.subplots(figsize=(5,5), dpi=200, facecolor='white')
fs          = 16    # font size
UpperCas    = 34    # upper cascade wave number
LowerCas    = 80    # lower cascade wave number
DrivScale   = 3     # driving scale wave number
DissScale   = 220   # dissipation scale wave number

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

azi_comb = np.array((Blue2DSpectra_azi,Green2DSpectra_azi,Red2DSpectra_azi))

azi_k_aver              = np.mean(azi_comb, axis=0)  # average along the 0 axis
azi_k_channel_var       = np.var(azi_comb, axis=0)   # the channel variane
azi_k_aver_var          = np.hypot(Blue_var,Green_var,Red_var) # the azimuthal averaging variance
azi_k_aver_var_upper    = 10 ** ( np.log10(azi_k_aver)+(1/np.log(10)*np.divide(azi_k_aver_var,azi_k_aver) ) )
azi_k_aver_var_lower    = 10 ** ( np.log10(azi_k_aver)-(1/np.log(10)*np.divide(azi_k_aver_var,azi_k_aver) ) )

ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
ax.plot(k, azi_k_aver_var_upper,color='blue',linewidth=0.75,ls=':',label=r'$1\sigma$ variation')
ax.plot(k, azi_k_aver_var_lower,color='blue',linewidth=0.75,ls=':')
im1 = ax.plot(k, azi_k_aver,color='black',label='Averaged power spectrum')
ax.set_xlabel(r'$|\mathbf{k}|$',size=fs,labelpad=-0.5)
ax.set_ylabel(r'$\left\langle\mathcal{P}(|\mathbf{k}|) 2 \pi |\mathbf{k}| \right\rangle_{\theta}$',size=fs)
ax.axvline(x=UpperCas,color='red',ls='--')
ax.axvline(x=LowerCas,color='red',ls='--')
ax.annotate(r'$\ell_D$',xy=(DrivScale, 15),fontsize=fs+2,color='red')
ax.annotate(r'$\ell_\nu$',xy=(DissScale, 0.009),fontsize=fs+2,color='red')
ax.legend(prop={'size': 12})

#plt.savefig('Figure4.png',dpi=200)
plt.close()

f, ax = plt.subplots(1,2,figsize=(6,2), dpi=200, facecolor='white')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

ax[0].set_xscale("log", nonposx='clip')
ax[0].set_yscale("log", nonposy='clip')
ax[0].plot(k, azi_k_aver_var_upper*k**(5.0/3),color='blue',linewidth=0.75,ls=':')
ax[0].plot(k, azi_k_aver_var_lower*k**(5.0/3),color='blue',linewidth=0.75,ls=':')
im2 = ax[0].plot(k, azi_k_aver*k**(5.0/3),color='black')
ax[0].set_xlabel(r'$|\mathbf{k}|$',size=fs,labelpad=-0.5)
ax[0].set_ylabel(r'$\left\langle\mathcal{P}(|\mathbf{k}|) 2 \pi |\mathbf{k}| \right\rangle_{\theta}/|\mathbf{k}|^{-5/3}$',size=fs)
ax[0].axvline(x=UpperCas,color='red',ls='--')
ax[0].axvline(x=LowerCas,color='red',ls='--')
ax[0].annotate(r'$\ell_D$',xy=(DrivScale-0.5, 33),fontsize=fs+2,color='red')
ax[0].annotate(r'$\ell_\nu$',xy=(DissScale, 56.5),fontsize=fs+2,color='red')

ax[1].set_xscale("log", nonposx='clip')
ax[1].set_yscale("log", nonposy='clip')
ax[1].plot(k, azi_k_aver_var_upper*k**2,color='blue',linewidth=0.75,ls=':')
ax[1].plot(k, azi_k_aver_var_lower*k**2,color='blue',linewidth=0.75,ls=':')
im2 = ax[1].plot(k, azi_k_aver*k**2,color='black')
ax[1].set_xlabel(r'$|\mathbf{k}|$',size=fs,labelpad=-0.5)
ax[1].set_ylabel(r'$\left\langle\mathcal{P}(|\mathbf{k}|) 2 \pi |\mathbf{k}| \right\rangle_{\theta}/|\mathbf{k}|^{-2}$',size=fs)
ax[1].axvline(x=UpperCas,color='red',ls='--')
ax[1].axvline(x=LowerCas,color='red',ls='--')
ax[1].annotate(r'$\ell_D$',xy=(DrivScale-0.5, 111),fontsize=fs+2,color='red')
ax[1].annotate(r'$\ell_\nu$',xy=(DissScale-5, 180),fontsize=fs+2,color='red')

#plt.savefig('Figure5.png',dpi=200)
plt.close()


# Slope Calculation
############################################################################################################################################

k_fit       = k[UpperCas:LowerCas]
Power_fit   = azi_k_aver[UpperCas:LowerCas]

# Slope in the pseudo energy cascade
slope, intercept, x, y = linear_regression(k_fit,Power_fit,k_fit)

# Uncertainty in the slope
abs(slope) * ( np.mean( azi_k_aver_var[UpperCas:LowerCas] ) / np.mean(Power_fit) )
