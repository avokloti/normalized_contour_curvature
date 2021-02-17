# -*- coding: utf-8 -*-
"""4. Viewpoint and Illumination

Andrew Marantan, Irina Tolkova, and Lakshminarayanan Mahadevan (2020)
"""

# try to reproduce curvature calculations of Andrew's code
import numpy as np
import cv2
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.manifold import MDS


def contourCurvature(image, rho, numBins, background_threshold = 5, plot=False, cutoff=True, add_padding=False):
    # --- SETTINGS --- #
    
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
    mask = f < background_threshold
    kappa[mask] = np.nan

    # --- Calculate Histogram --- #

    # compute histogram for well-defined curavtures #
    kappaHist = np.histogram(np.ndarray.flatten(kappa), binEdges)

    # include flat regions #
    #flatCounts = np.sum(np.ndarray.flatten(kappa) == -dKappa)
    flatCounts = 0

    #kappaHist = (np.insert(kappaHist[0], 0, flatCounts), np.insert(kappaHist[1], 0, -dKappa))

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

def processImage(filename):
    # process image
    image = cv2.imread(filename)
    image = image.astype('float')
    image = image[:,:,0]
    image[image < 25] = image[image < 25] * 0
    return image


# ---------------------------------------------------------------- #
# Set the category variable to either 'Viewpoint' or 'Illumination'!
category = 'Viewpoint'
#category = 'Illumination'
# ---------------------------------------------------------------- #

# find each object in the directory
workdir = '../stimuli/ALOI/' + category + '/'

all_objects = [9, 96, 404, 208, 155, 7]
object_names = ['Shoe', 'String', 'Cat', 'Clock', 'Shuttlecock', 'Seashell']
num_objects = len(all_objects)
num_samples = 15

# set random seed
np.random.seed(4)

labels = []
filenames = []

# for each object...
for dataset_index in all_objects:
    # get all filenames
    all_views = os.listdir(workdir + str(dataset_index) + '/')

    # choose ten images
    indices = np.random.choice(np.arange(len(all_views)), size=(num_samples), replace=False)

    # find their filenames
    object_filenames = [workdir + str(dataset_index) + '/' + all_views[i] for i in indices]

    # add to lists
    filenames = filenames + object_filenames
    labels = labels + [dataset_index] * num_samples

# for each filename, calculate NCC
num_images = len(filenames)
rho = 0.04
num_bins = 101

histogram_list = np.zeros((num_images, num_bins))

for i in range(num_images):
    image = processImage(filenames[i])

    # calculate NCC
    kappa, kappaDist, bins, f = contourCurvature(image, rho, num_bins)

    # put in histogram_list
    histogram_list[i,:] = kappaDist/sum(kappaDist)

if (category == 'Viewpoint'):
    example_inds = [[2, 3, 12], [0, 1, 2], [1, 2, 4], [1, 4, 6], [0, 1, 2], [0, 1, 4]]
elif (category == 'Illumination'):
    example_inds = [[0, 7, 3], [0, 2, 4], [1, 6, 11], [0, 1, 3], [0, 1, 2], [2, 5, 4]]

# make a figure to demonstrate the dataset
plt.figure(figsize=(16, 6))
for i in range(6):
    # take three examples
    for j in range(3):
        ind = num_samples * i + example_inds[i][j]
        image = processImage(filenames[ind])

        plt.subplot(3, 6, 6 * j + i + 1)
        plt.imshow(image, cmap='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

plt.tight_layout()


# ---- Calculate JS Divergence and Perform MDS --- #

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return np.sum(np.where(p != 0, p * np.log(p / m), 0)) + np.sum(np.where(q != 0, q * np.log(q / m), 0))

dist_mat_js = np.zeros((num_images, num_images))
for i in range(num_images):
  for j in range(num_images):
    dist_mat_js[i, j] = jensen_shannon_divergence(histogram_list[i,:], histogram_list[j,:])

colors = ['r', 'g', 'b', 'c', 'k', 'purple']

mds = MDS(n_components=2, n_init=50, dissimilarity='precomputed')
output_2 = mds.fit_transform(dist_mat_js)

# plot
plt.figure(figsize=(8, 8))
for i in range(num_objects):
  plt.scatter(output_2[num_samples * i:num_samples*(i+1),0], output_2[num_samples * i:num_samples*(i+1),1], color = colors[i])

plt.legend(object_names, fontsize=16)
plt.xlabel('Dimension 1', fontsize=16)
plt.ylabel('Dimension 2', fontsize=16)
plt.title('MDS Applied To %s' % (category), fontsize=18)
plt.ylim([-0.15, 0.22])

plt.show()