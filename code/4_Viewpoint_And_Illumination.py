"""4. Viewpoint and Illumination

This script examines robustness of NCC across varying viewpoint and illumination conditions by analysis of the Amsterdam Library of Object Images (ALOI).
"""

import numpy as np
import cv2
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

# import functions
from functions import *

# ---------------------------------------------------------------- #
# Set the category variable to either 'Viewpoint' or 'Illumination'
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
    image = processImageALOI(filenames[i])

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
        image = processImageALOI(filenames[ind])

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
