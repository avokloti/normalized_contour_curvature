"""3. Clustering of Text, Cartoon Faces, and Tetris Shapes

This script characterizes a dataset of alpha-numeric characters, cartoon faces, and
Tetris shapes through principal component analysis over normalized contour curvature.
"""

import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# import functions
from functions import *

## ----- Prepare Data Paths ----- ##

# prepare outer directory
stimuli_dir = '../stimuli/SymbolsPatterns/'

# categories
categories = ['Faces', 'Helvetica', 'Tetris']

# find all filenames
face_files = os.listdir(stimuli_dir + categories[0])
helvetica_files = os.listdir(stimuli_dir + categories[1])
tetris_files = os.listdir(stimuli_dir + categories[2])

# full paths for all filenames
face_files = [stimuli_dir + categories[0] + "/" + path for path in face_files]
helvetica_files = [stimuli_dir + categories[1] + "/" + path for path in helvetica_files]
tetris_files = [stimuli_dir + categories[2] + "/" + path for path in tetris_files]

# collect all files
all_images = face_files + helvetica_files + tetris_files
num_images = len(all_images)

## ----- Calculate Histograms ----- ##

# set number of histogram bins
numBins = 51

# pixel value defining background
bg_threshold = 1

# determine rho values
rho = 0.018

# prepare list to store values
histogram_list = np.zeros((num_images, numBins))
for i in range(num_images):
    # read in image and convert to grayscale float
    image = processImageBMP(all_images[i])

    # calculate NCC
    kappa, kappaDist, bins, f = contourCurvature(image, rho, numBins, bg_threshold)

    # put in histogram_list
    histogram_list[i,:] = kappaDist

    # print
    if (i % 10 == 0):
        print('--- Finished calculation for image %d/%d' % (i, num_images))


## ----- Plot Example Images From Each Category ----- ##

plt.rcParams.update({'font.size': 10})

names = ['Faces/faceL00.bmp', 'Faces/faceLT.bmp', 'Tetris/MN7.bmp', 'Tetris/MNW.bmp', 'Helvetica/NC.bmp', 'Helvetica/NW.bmp']
names = [stimuli_dir + name for name in names]

plt.figure(figsize=(24, 9))
for i in range(len(names)):
    p1 = all_images[all_images.index(names[i])]

    image = processImageBMP(p1)
    kappa, kappaDist, bins, f, paddedImage = contourCurvature(image, rho, numBins, bg_threshold, False, False, True)

    plt.subplot(3, len(names), i + 1)
    plt.imshow(paddedImage, cmap='gray_r')
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(3, len(names), i + len(names) + 1)
    plt.imshow(kappa)
    plt.axis('off')
    plt.title('NCC')
    plt.colorbar()
    plt.axis('equal')

    plt.subplot(3, len(names), i + 2 * len(names) + 1)
    plt.plot(bins, kappaDist, color='royalblue')
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$P(\kappa)$')
    plt.title('Normalized Contour Curvature Distribution')

    plt.tight_layout()


## ----- Calculate SVD ----- ##

X = histogram_list.T
u, s, vh = np.linalg.svd(X)

## ----- Plot Singular Values, Principal Components, And Coefficients ----- ##

plt.rcParams.update({'font.size': 14})

# plot first ten singular values
plt.figure()
plt.plot(s[0:10], '*-')
plt.title('First 10 Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value')

# plot first three singular components
fig = plt.figure(figsize=(6, 4))

plt.plot(bins, -u[:, 0], color='r', label = 'Component 1')
plt.plot(bins, -u[:, 1], color='purple', label = 'Component 2')
plt.plot(bins, u[:, 2], color='darkorange', label = 'Component 3')
plt.title('First Three Principal Components', fontsize=16)
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$')
plt.ylabel(r'Probability Density $P(\hat{\kappa})$')
plt.legend()

# plot weights for first three singular components
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['aqua', 'b', 'lime', 'r']

ax.scatter(vh[0, 0:26], vh[1,0:26], vh[2,0:26], color=colors[0], label = categories[0])
ax.scatter(vh[0, 26:52], vh[1,26:52], vh[2,26:52], color=colors[1], label = categories[1])
ax.scatter(vh[0, 52:78], vh[1,52:78], vh[2,52:78], color=colors[2], label = categories[2])

ax.set_xlabel('Component 1', fontsize=16)
ax.set_ylabel('Component 2', fontsize=16)
ax.set_zlabel('Component 3', fontsize=16)
ax.view_init(elev=40., azim=35)
ax.tick_params(axis='both', which='major', pad=-10)
plt.title('Weights for First Three Singular Components, by Category', fontsize=18)
plt.legend(loc='center right', fontsize=16)

plt.tight_layout()

## ----- Plot Coefficients with 2D Projection ----- ##

colors = ['c', 'b', 'g', 'r']

comp0 = -1 * np.squeeze(np.reshape(u[:, 0], (1, numBins)) @ histogram_list.T)
comp1 = -1 * np.squeeze(np.reshape(u[:, 1], (1, numBins)) @ histogram_list.T)
comp2 = np.squeeze(np.reshape(u[:, 2], (1, numBins)) @ histogram_list.T)

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.scatter(comp0[0:26], comp1[0:26], color=colors[0], label = categories[0])
plt.scatter(comp0[26:52], comp1[26:52], color=colors[1], label = categories[1])
plt.scatter(comp0[52:78], comp1[52:78], color=colors[2], label = categories[2])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(comp0[0:26], comp2[0:26], color=colors[0], label = categories[0])
plt.scatter(comp0[26:52], comp2[26:52], color=colors[1], label = categories[1])
plt.scatter(comp0[52:78], comp2[52:78], color=colors[2], label = categories[2])
plt.xlabel('Component 1')
plt.ylabel('Component 3')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(comp1[0:26], comp2[0:26], color=colors[0], label = categories[0])
plt.scatter(comp1[26:52], comp2[26:52], color=colors[1], label = categories[1])
plt.scatter(comp1[52:78], comp2[52:78], color=colors[2], label = categories[2])
plt.xlabel('Component 2')
plt.ylabel('Component 3')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.show()


"""3. Clustering of Wavy Against Straight Stimuli

This script characterizes a dataset of abstract stimuli divided into "wave", "bead", and "straight" categories.
"""

# prepare outer directory
stimuli_dir = '../stimuli/SymbolsPatterns/'

categories = ['Beads_Straight', 'Waves_Straight', 'Straight']

bead_files = os.listdir(stimuli_dir + categories[0])
wave_files = os.listdir(stimuli_dir + categories[1])
straight_files = os.listdir(stimuli_dir + categories[2])

bead_files = [stimuli_dir + categories[0] + "/" + path for path in bead_files]
wave_files = [stimuli_dir + categories[1] + "/" + path for path in wave_files]
straight_files = [stimuli_dir + categories[2] + "/" + path for path in straight_files]

all_images = bead_files + wave_files + straight_files
num_images = len(all_images)

## ----- Calculate Histograms ----- ##

# set number of histogram bins
numBins = 51

# determine rho values
rho = 0.018

# prepare list to store values
histogram_list = np.zeros((num_images, numBins))
for i in range(num_images):
    # read in image and convert to grayscale float
    image = processImageBMP(all_images[i])

    # calculate NCC
    kappa, kappaDist, bins, f = contourCurvature(image, rho, numBins, patterns=True)

    # put in histogram_list
    histogram_list[i,:] = kappaDist

    # print
    if (i % 10 == 0):
        print('--- Finished calculation for image %d/%d' % (i, num_images))

## ----- Plot Example Images From Each Category ----- ##

plt.rcParams.update({'font.size': 10})

names = ['Beads_Straight/Beads_Straight_01.jpg', 'Beads_Straight/Beads_Straight_08.jpg', 'Waves_Straight/Waves_Straight_01.jpg', 'Waves_Straight/Waves_Straight_08.jpg', 'Straight/Straight_01.jpg', 'Straight/Straight_08.jpg']
names = [stimuli_dir + name for name in names]

plt.figure(figsize=(24, 9))
for i in range(len(names)):
    p1 = all_images[all_images.index(names[i])]

    #rho = 0.018
    image = processImageBMP(p1)
    kappa, kappaDist, bins, f, paddedImage = contourCurvature(image, rho, numBins, bg_threshold, False, False, True, patterns=True)

    plt.subplot(3, len(names), i + 1)
    plt.imshow(paddedImage, cmap='gray_r')
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(3, len(names), i + len(names) + 1)
    plt.imshow(kappa)
    plt.axis('off')
    plt.title('NCC')
    plt.colorbar()
    plt.axis('equal')

    plt.subplot(3, len(names), i + 2 * len(names) + 1)
    plt.plot(bins, kappaDist, color='royalblue')
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$P(\kappa)$')
    plt.title('Normalized Contour Curvature Distribution')

    plt.tight_layout()

## ----- Calculate SVD ----- ##

X = histogram_list.T
u, s, vh = np.linalg.svd(X)
u = -u
vh = -vh

## ----- Plot Singular Values, Principal Components, And Coefficients ----- ##

plt.rcParams.update({'font.size': 14})

category_names = ['Beads', 'Waves', 'Straight']

plt.figure()
plt.plot(s[0:10], '*-')
plt.title('First 10 Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value')

# plot first three singular components
fig = plt.figure(figsize=(6, 4))

plt.plot(bins, u[:, 0], 'r', label = 'Component 1')
plt.plot(bins, -u[:, 1], 'darkorange', label = 'Component 2')
plt.plot(bins, u[:, 2], 'purple', label = 'Component 3')
plt.title('First Three Principal Components', fontsize=16)
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$')
plt.ylabel(r'Probability Density $P(\hat{\kappa})$')
plt.legend()

# plot weights for first three singular components
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['aqua', 'lime', 'b']

ax.scatter(vh[0, 0:13], -vh[1,0:13], vh[2,0:13], color=colors[0], label = category_names[0], depthshade=False)
ax.scatter(vh[0, 13:26], -vh[1,13:26], vh[2,13:26], color=colors[1], label = category_names[1], depthshade=False)
ax.scatter(vh[0, 26:39], -vh[1,26:39], vh[2,26:39], color=colors[2], label = category_names[2], depthshade=False)

ax.set_xlabel('Component 1', fontsize=16)
ax.set_ylabel('Component 2', fontsize=16)
ax.set_zlabel('Component 3', fontsize=16)
ax.view_init(elev=30., azim=40)
ax.tick_params(axis='both', which='major', pad=-10)
plt.title('Weights for First Three Components', fontsize=18)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.8))

plt.tight_layout()

# plot weights for first three singular components
comp0 = np.squeeze(np.reshape(u[:, 0], (1, numBins)) @ histogram_list.T)
comp1 = np.squeeze(np.reshape(u[:, 1], (1, numBins)) @ histogram_list.T)
comp2 = np.squeeze(np.reshape(u[:, 2], (1, numBins)) @ histogram_list.T)

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.scatter(comp0[0:13], comp1[0:13], color=colors[0], label = categories[0])
plt.scatter(comp0[13:26], comp1[13:26], color=colors[1], label = categories[1])
plt.scatter(comp0[26:39], comp1[26:39], color=colors[2], label = categories[2])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(comp0[0:13], comp2[0:13], color=colors[0], label = categories[0])
plt.scatter(comp0[13:26], comp2[13:26], color=colors[1], label = categories[1])
plt.scatter(comp0[26:39], comp2[26:39], color=colors[2], label = categories[2])
plt.xlabel('Component 1')
plt.ylabel('Component 3')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(comp1[0:13], comp2[0:13], color=colors[0], label = categories[0])
plt.scatter(comp1[13:26], comp2[13:26], color=colors[1], label = categories[1])
plt.scatter(comp1[26:39], comp2[26:39], color=colors[2], label = categories[2])
plt.xlabel('Component 2')
plt.ylabel('Component 3')
plt.title('Singular Component Weights, by Category')
plt.legend()

plt.show()
