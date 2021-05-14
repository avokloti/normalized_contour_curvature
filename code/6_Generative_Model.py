# -*- coding: utf-8 -*-
"""6. Generative Model

This notebook implements the generative model for constructing artificial images with NCC statistics matching a given distribution.

Andrew Marantan, Irina Tolkova, and Lakshminarayanan Mahadevan (2021)
"""

import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm


def contourCurvature(image, numBins, plot=False):
    # --- SETTINGS --- #
    
    # histogram bin edges
    kappaMin = -1
    kappaMax = 1
    dKappa = 2/numBins
    binEdges = np.linspace(kappaMin, kappaMax, numBins+1)

    bins = binEdges[0:-1] + dKappa/2

    # --- Retrieve Image Properties --- #

    # obtain image dimensions #
    [numRows, numCols] = image.shape
    imageLength = np.max((numRows, numCols))
    
    # set pixel spacing #
    dx = 1 / imageLength

    # --- Calculate Contour Curvature --- #

    # shift image #
    f_N = np.roll(image, -1, 0)
    f_E = np.roll(image, -1, 1)
    f_S = np.roll(image, 1, 0)
    f_W = np.roll(image, 1, 1)

    f_NE = np.roll(np.roll(image, -1, 0), -1, 1)
    f_SE = np.roll(np.roll(image, 1, 0), -1, 1)
    f_SW = np.roll(np.roll(image, 1, 0), 1, 1)
    f_NW = np.roll(np.roll(image, -1, 0), 1, 1)

    # construct derivatives #
    f_x = (f_E - f_W)/(2 * dx)
    f_y = (f_N - f_S)/(2 * dx)
    f_xx = (f_E - 2*image + f_W)/(dx**2)
    f_yy = (f_N - 2*image + f_S)/(dx**2)
    f_xy = (f_NE - f_SE - f_NW + f_SW)/(4 * dx**2)

    # compute contour curvature #
    kappa = f_y**2 * f_xx + f_x**2 * f_yy - 2 * f_x * f_y * f_xy
    kappa = kappa * (f_x**2 + f_y**2)**(-3/2)

    # convert to normalized curvature #
    kappa = -kappa / (2 + np.abs(kappa))

    # set background pixels to NaN #
    mask = np.isnan(image)
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

    return kappa, kappaDist, bins

def generatePatch(patch_X, patch_Y, bound_X, bound_Y, boundVals, xi):
    patchSize = patch_X.shape[0]
    
    numPatch = patch_X.shape[0] * patch_X.shape[1]
    numBound = bound_X.shape[0] * bound_X.shape[1]
    
    patch_X = np.reshape(patch_X, (numPatch, 1))
    patch_Y = np.reshape(patch_Y, (numPatch, 1))

    bound_X = np.reshape(bound_X, (numBound, 1))
    bound_Y = np.reshape(bound_Y, (numBound, 1))

    pos_X = np.concatenate((patch_X, bound_X))
    pos_Y = np.concatenate((patch_Y, bound_Y))

    # construct a distance matrix
    R2 = np.zeros((numPatch + numBound, numPatch + numBound))
    for ii in range(numPatch + numBound):
        for jj in range(numPatch + numBound):
            R2[ii, jj] = (pos_X[ii, 0] - pos_X[jj, 0])**2 + (pos_Y[ii, 0] - pos_Y[jj, 0])**2
    
    # calculate covariance matrix and mean vector #
    Sigma = np.exp( - R2 / (2 * xi**2) )
    patchMean = np.zeros((numPatch, 1))
    
    if (numBound > 0):
        Sigma_pp = Sigma[0:numPatch, 0:numPatch]
        Sigma_pb = Sigma[0:numPatch, numPatch:]
        Sigma_bb = 10**(-4) * np.eye(numBound) +  Sigma[numPatch:, numPatch:]

        # calculate patch mean #
        lstsq_output = np.linalg.lstsq(Sigma_bb, boundVals)
        patchMean = Sigma_pb @ lstsq_output[0]

        # calculate conditional covariance matrix #
        lstsq_output = np.linalg.lstsq(Sigma_bb, Sigma_pb.T)
        Sigma = Sigma_pp - Sigma_pb @ lstsq_output[0]
    
    # calculate matrix square root #
    T = np.real(sqrtm(Sigma))

    # generate uncorrelated Gaussian rvs #
    rawGaussians = np.random.normal(0,  1, (T.shape[0], 1))

    # correlate variables #
    patch = np.transpose(T) @ rawGaussians + patchMean
    patch = np.reshape(patch, (patchSize, patchSize))
    return patch

def drawXi(xiVec, xiCDF):
    # draw s1 and s2 in [0,1]
    s1 = np.random.rand()
    s2 = np.random.rand()
    
    # find xi bin
    xiIndex = np.nonzero(xiCDF > s1)[0][0]
    
    # draw xi from within bin
    return xiVec[xiIndex] + (s2 - 1/2) * (xiVec[1] - xiVec[0])


def generateImage(xiVec, xiDist, imageSize, patchSize):
  xiCDF = np.cumsum(xiDist)

  # extract image size parameters (IT: assumes square image)
  numRows = imageSize[0] - np.mod(imageSize[0], patchSize)
  numCols = imageSize[1] - np.mod(imageSize[1], patchSize)

  # boundary setting
  boundOffset = patchSize

  # instantiate image with all NaNs
  image = np.empty((numRows, numCols))
  image[:] = np.nan

  # instantiate position arrays
  xVec = (np.arange(numCols) - 1/2 * np.ones(numCols)) / numCols
  yVec = (np.arange(numRows) - 1/2 * np.ones(numRows)) / numCols
  [X, Y] = np.meshgrid(xVec, yVec)

  for ii in range(numRows // patchSize):
      for jj in range(numCols // patchSize):
          # obtain correlation length
          xi = drawXi(xiVec, xiCDF)

          # extract patch coordinates
          # X and Y indices #
          minRow = ii * patchSize
          maxRow = (ii+1) * patchSize
          minCol = jj * patchSize
          maxCol = (jj + 1) * patchSize

          # patch pixel positions #
          patch_X = X[minRow:maxRow, minCol:maxCol]
          patch_Y = Y[minRow:maxRow, minCol:maxCol]

          # extract boundary information #
          # initialize boundary arrays #
          bound_W_X = []
          bound_W_Y = []
          bound_W_Vals = []

          bound_N_X = []
          bound_N_Y = []
          bound_N_Vals = []

          # extract northern boundary properties #
          if (ii > 0):
              bound_N_rows_min = np.max((0, minRow - boundOffset))
              bound_N_rows_max = minRow
              bound_N_cols_min = np.max((0, minCol-boundOffset))
              bound_N_cols_max = np.min((maxCol+boundOffset, numCols))

              bound_N_rows = np.arange(bound_N_rows_min, bound_N_rows_max)
              bound_N_cols = np.arange(bound_N_cols_min, bound_N_cols_max)

              bound_N_X = X[bound_N_rows_min:bound_N_rows_max, bound_N_cols_min:bound_N_cols_max]
              bound_N_Y = Y[bound_N_rows_min:bound_N_rows_max, bound_N_cols_min:bound_N_cols_max]
              bound_N_Vals = image[bound_N_rows_min:bound_N_rows_max, bound_N_cols_min:bound_N_cols_max]
          
          # extract western boundary properties #
          if (jj > 0):
              bound_W_rows_min = minRow
              bound_W_rows_max = maxRow
              bound_W_cols_min = np.max((0, minCol - boundOffset))
              bound_W_cols_max = minCol

              bound_W_rows = np.arange(bound_W_rows_min, bound_W_rows_max)
              bound_W_cols = np.arange(bound_W_cols_min, bound_W_cols_max)

              bound_W_X = X[bound_W_rows_min:bound_W_rows_max, bound_W_cols_min:bound_W_cols_max]
              bound_W_Y = Y[bound_W_rows_min:bound_W_rows_max, bound_W_cols_min:bound_W_cols_max]
              bound_W_Vals = image[bound_W_rows_min:bound_W_rows_max, bound_W_cols_min:bound_W_cols_max]
          
          # set boundary arrays
          bound_W_X_size = 0
          bound_W_Y_size = 0
          bound_N_X_size = 0
          bound_N_Y_size = 0
          bound_W_Vals_size = 0
          bound_N_Vals_size = 0

          if (len(bound_W_X) != 0):
              bound_W_X_size = bound_W_X.size
          if (len(bound_W_Y) != 0):
              bound_W_Y_size = bound_W_Y.size
          if (len(bound_N_X) != 0):
              bound_N_X_size = bound_N_X.size
          if (len(bound_N_Y) != 0):
              bound_N_Y_size = bound_N_Y.size
          if (len(bound_W_Vals) != 0):
              bound_W_Vals_size = bound_W_Vals.size
          if (len(bound_N_Vals) != 0):
              bound_N_Vals_size = bound_N_Vals.size
          
          bound_X = np.concatenate((np.reshape(bound_W_X, (bound_W_X_size, 1)), np.reshape(bound_N_X, (bound_N_X_size, 1))))
          bound_Y = np.concatenate((np.reshape(bound_W_Y, (bound_W_Y_size, 1)), np.reshape(bound_N_Y, (bound_N_Y_size, 1))))
          boundVals = np.concatenate((np.reshape(bound_W_Vals, (bound_W_Vals_size, 1)), np.reshape(bound_N_Vals, (bound_N_Vals_size, 1))))
          
          # generate patch given boundary values #
          patch = generatePatch(patch_X, patch_Y, bound_X, bound_Y, boundVals, xi)

          # add patch to image #
          image[minRow:maxRow, minCol:maxCol] = patch
  
  return image


## --- Test Generator --- ##

# set image size
imageSize = (50, 50)
# set patch size
patchSize = 5

# curvature settings #
numBins = 101
dKappa = 2/numBins
kappaBinEdges = np.linspace(-1, 1, numBins+1)
kappaBins = kappaBinEdges[0:-1] + dKappa/2

# load means of distribution for all four groups
category_filenames = {'Large Animate': 'big_animate_mean_ncc',
                      'Small Animate': 'small_animate_mean_ncc',
                      'Large Inanimate': 'big_inanimate_mean_ncc',
                      'Small Inanimate': 'small_inanimate_mean_ncc'}

calculation_dir = '../calculations'

# load in:
group_means = {}
xi_dists = {}
for cat in category_filenames.keys():
    with open(calculation_dir + '/' + category_filenames[cat] + '.pickle', 'rb') as pickle_file:
        group_mean = pickle.load(pickle_file)

    with open(calculation_dir + '/xi_dist_' + category_filenames[cat] + '.pickle', 'rb') as pickle_file:
        xi_dists[cat] = pickle.load(pickle_file)

with open(calculation_dir + '/xi_vec.pickle', 'rb') as pickle_file:
    xi_vec = pickle.load(pickle_file)


# initialize data arrays
images = {}
cutImages = {}
kappaImages = {}
kappaHists = {}

# for each category, solve for distribution and plot
for cat in category_filenames.keys():
    # generate images #
    images[cat] = generateImage(xi_vec, xi_dists[cat][0], imageSize, patchSize)

    # cut off values below threshold #
    images[cat] = images[cat]/np.std(images[cat])
    cutImage = images[cat]
    cutImage[images[cat] < -1] = np.nan
    
    # calculate NCC image and histogram #
    kappaImage, kappaHist, bins = contourCurvature(cutImage, numBins)
    
    # store data #
    cutImages[cat] = cutImage
    kappaImages[cat] = kappaImage
    kappaHists[cat] = kappaHist


# generate images #
fig, axs = plt.subplots(4, 3, figsize=(8, 11))

for n, cat in enumerate(category_filenames.keys()):
    plt.subplot(4, 3, 3 * n + 1)
    image_plot = np.nan_to_num(images[cat], copy=True, nan=np.nanmin(images[cat]))
    image_plot = image_plot/np.nanstd(images[cat])
    pcol = plt.pcolor(image_plot, cmap='gray_r', linewidth=0, rasterized=True)
    pcol.set_edgecolor('face')
    axs[n, 0].set_aspect('equal', 'box')
    if (n == 0):
        plt.title('Generated Image', fontsize=16)

    plt.ylabel(cat, fontsize=16)
    axs[n, 0].set_yticklabels([])
    axs[n, 0].set_yticks([])
    axs[n, 0].set_xticklabels([])
    axs[n, 0].set_xticks([])
    plt.axis('equal')

    plt.subplot(4, 3, 3 * n + 2)
    plt.contour(image_plot)
    if (n == 0):
        plt.title('Contour Plot', fontsize=16)
    plt.axis('equal')

    plt.subplot(4, 3, 3 * n + 3)
    plt.pcolor(kappaImages[cat])
    axs[n, 1].set_aspect('equal')
    plt.axis('equal')
    if (n == 0):
        plt.title('NCC Image', fontsize=16)

plt.tight_layout()
plt.show()