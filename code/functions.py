# -*- coding: utf-8 -*-
"""
This file includes functions for image processing and contour curvature calculation.
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

def contourCurvature(image, rho, numBins, background_threshold=5, plot=False, cutoff=True, add_padding=False, patterns=False):
    # --- Settings --- #
    
    # histogram bin edges
    dKappa = 2/numBins
    kappaMin = -1
    kappaMax = 1
    binEdges = np.linspace(kappaMin, kappaMax, numBins+1)
    
    #bins = np.hstack((kappaMin, binEdges[1:-1] + dKappa/2, kappaMax))
    bins = binEdges[0:-1] + dKappa/2
    
    # --- Retrieve Image Properties --- #
    
    # obtain image dimensions #
    [numRows, numCols] = image.shape
    imageLength = np.max((numRows, numCols))
    
    # set pixel spacing #
    dx = 1 / imageLength
    
    # set filter scale #
    sigma = rho * imageLength
    
    # --- Apply Gaussian Filter --- #
    
    # precompute filter sized based on Matlab standard #
    Q = int(8 * np.ceil(2*sigma) + 1)
    
    # pad image to ensure nothing is lost during filtering #
    padding = int(np.floor(Q/2))
    paddedImage = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    # filter
    f = cv2.GaussianBlur(paddedImage, (Q, Q), sigma, sigma)
    
    # --- Calculate Contour Curvature --- #
    
    # shift image #
    f_N = np.roll(f, -1, 0)
    f_E = np.roll(f, -1, 1)
    f_S = np.roll(f, 1, 0)
    f_W = np.roll(f, 1, 1)
    
    f_NE = np.roll(np.roll(f, -1, 0), -1, 1)
    f_SE = np.roll(np.roll(f, 1, 0), -1, 1)
    f_SW = np.roll(np.roll(f, 1, 0), 1, 1)
    f_NW = np.roll(np.roll(f, -1, 0), 1, 1)
    
    # construct derivatives #
    f_x = (f_E - f_W)/(2 * dx)
    f_y = (f_N - f_S)/(2 * dx)
    f_xx = (f_E - 2*f + f_W)/(dx**2)
    f_yy = (f_N - 2*f + f_S)/(dx**2)
    f_xy = (f_NE - f_SE - f_NW + f_SW)/(4 * dx**2)
    
    # compute contour curvature #
    kappa = f_y**2 * f_xx + f_x**2 * f_yy - 2 * f_x * f_y * f_xy
    kappa = kappa * (f_x**2 + f_y**2)**(-3/2)
    
    # convert to normalized curvature #
    kappa = -kappa / (2 + np.abs(kappa))
    
    # set flat patches to - dKappa #
    kappa[np.isnan(kappa)] = -dKappa
    
    # set background pixels to NaN #
    if (patterns): # if we are analyzing discrete patterns
        mask = np.ones(paddedImage.shape)
        mask[padding:-padding, padding:-padding] = 0 * mask[padding:-padding, padding:-padding]
        kappa[mask == 1] = np.nan
    else: # if we are analyzing smooth images
        mask = f < background_threshold
        kappa[mask] = np.nan

    # --- Calculate Histogram --- #
    
    # compute histogram for well-defined curavtures #
    kappaHist = np.histogram(np.ndarray.flatten(kappa), binEdges)
    
    # convert histogram to probability distribution #
    kappaDist = kappaHist[0] / np.sum( dKappa * kappaHist[0])

    if (plot):
        plt.figure(figsize=(16, 5))
        
        plt.subplot(1, 2, 1)
        plt.pcolor(kappa)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title('Normalized Contour Curvature of Image')
        
        plt.subplot(1, 2, 2)
        plt.plot(bins, kappaDist)
        plt.title('Probability Density of NCC')
        plt.xlabel('Normalized Contour Curvature')
        plt.ylabel('Probability Density')
    
    if (cutoff):
        return kappa[padding:-padding, padding:-padding], kappaDist, bins, f[padding:-padding, padding:-padding]
    elif (add_padding):
        return kappa, kappaDist, bins, f, paddedImage
    else:
        return kappa, kappaDist, bins, f


def processImage(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')
    image = 255 - image
    return image

def processImageBlackBG(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')
    return image

def processImageBMP(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')
    image = image - image[0,  0]
    return image

def processImageFaceBMP(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')
    image = 255 - image
    return image

def processImageALOI(filename):
    # process image
    image = cv2.imread(filename)
    image = image.astype('float')
    image = image[:,:,0]
    image[image < 25] = image[image < 25] * 0
    return image
