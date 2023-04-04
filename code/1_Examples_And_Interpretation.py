"""1. Examples And Interpretation of Normalized Contour Curvature

This script shows the application and interpretation of normalized contour
curvature (NCC) when applied to a dataset of object images, varying in animacy and size.
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import pickle

# import functions
from functions import *

## ----- Prepare Data Paths and Labels ----- ##

# set path to outer directory
stimuli_dir = '../Stimuli/AnimacySize'

categories = ['/Big-Animate/', '/Small-Animate/', '/Big-Inanimate/', '/Small-Inanimate/']

# collect all filenames
big_animate = os.listdir(stimuli_dir + categories[0])
small_animate = os.listdir(stimuli_dir + categories[1])
big_inanimate = os.listdir(stimuli_dir + categories[2])
small_inanimate = os.listdir(stimuli_dir + categories[3])

# add outer directory to all filenames
big_animate = [stimuli_dir + categories[0] + path for path in big_animate]
small_animate = [stimuli_dir + categories[1] + path for path in small_animate]
big_inanimate = [stimuli_dir + categories[2] + path for path in big_inanimate]
small_inanimate = [stimuli_dir + categories[3] + path for path in small_inanimate]

# take all images
all_images = big_animate + small_animate + big_inanimate + small_inanimate
num_images = len(all_images)

# set labels to differentiate either animate/inanimate or big/small
labels_animacy = [1] * int(num_images/2) + [0] * int(num_images/2)
labels_size = [1] * int(num_images/4) + [0] * int(num_images/4) + [1] * int(num_images/4) + [0] * int(num_images/4)

## ----- Calculate NCC For All Images ----- ##

# parameters
rho = 0.04
bg_threshold = 5
num_bins = 101

# make histogram bins
dKappa = 2/num_bins
kappaMin = -1
kappaMax = 1
binEdges = np.linspace(start=kappaMin, stop=kappaMax, num=num_bins+1)
bins = binEdges[0:-1] + dKappa/2

# prepare array to hold calculated histograms
histogram_list = np.zeros((num_images, num_bins))

for i in range(num_images):
    # get image
    image = processImage(all_images[i])

    # calculate NCC
    kappa, kappaDist, bins, f = contourCurvature(image, rho, numBins=num_bins)

    # put in list
    histogram_list[i,:] = kappaDist/(dKappa * sum(kappaDist))

    if (np.mod(i, 10) == 0):
        print('Completed image %d/%d.' % (i, num_images))

## ----- Calculate And Plot NCC Distributions Across Animacy/Size ----- ##

# animate
animate_histograms = histogram_list[0:120, :]
animate_mean = np.mean(animate_histograms, axis=0)
animate_sd = np.std(animate_histograms, axis=0)

# inanimate
inanimate_histograms = histogram_list[120:240, :]
inanimate_mean = np.mean(inanimate_histograms, axis=0)
inanimate_sd = np.std(inanimate_histograms, axis=0)

# large
big_histograms = histogram_list[0:60,:] + histogram_list[120:180,:]
big_mean = np.mean(big_histograms, axis=0)
big_sd = np.std(big_histograms, axis=0)

# small
small_histograms = histogram_list[60:120,:] + histogram_list[180:240,:]
small_mean = np.mean(small_histograms, axis=0)
small_sd = np.std(small_histograms, axis=0)

# plot means for both categories
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(8, 3)

ax[0].plot(bins, np.squeeze(animate_mean), label='Animate', color='C2')
ax[0].fill_between(bins, np.squeeze(animate_mean) - np.squeeze(animate_sd), np.squeeze(animate_mean) + np.squeeze(animate_sd), alpha=0.5, color='C2')
ax[0].plot(bins, np.squeeze(inanimate_mean), label='Inanimate', color='C3')
ax[0].fill_between(bins, np.squeeze(inanimate_mean) - np.squeeze(inanimate_sd), np.squeeze(inanimate_mean) + np.squeeze(inanimate_sd), alpha=0.5, color='C3')
ax[0].set_xlabel('Normalized Contour Curvature')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Distributions Across Animacy')
ax[0].legend()

ax[1].plot(bins, np.squeeze(big_mean), label='Large', color='C0')
ax[1].fill_between(bins, np.squeeze(big_mean) - np.squeeze(big_sd), np.squeeze(big_mean) + np.squeeze(big_sd), alpha=0.5, color='C0')
ax[1].plot(bins, np.squeeze(small_mean), label='Small', color='C1')
ax[1].fill_between(bins, np.squeeze(small_mean) - np.squeeze(small_sd), np.squeeze(small_mean) + np.squeeze(small_sd), alpha=0.5, color='C1')
ax[1].set_xlabel('Normalized Contour Curvature')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('Distributions Across Size')
ax[1].legend()

plt.tight_layout()

## --- print animate mean to file! ---

#plt.plot(bins, animate_mean)

#with open(stimuli_dir + '/animate_mean_aug15', 'wb') as pickle_file:
#    pickle.dump(animate_mean, pickle_file)

## ----- Calculate And Plot NCC Distributions Across Animacy/Size ----- ##

# plot means for both categories
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(10, 6)

colors = ['C2', 'C3', 'C0', 'C1']
titles = ['Large Animate NCC Distribution', 'Small Animate NCC Distribution',
          'Large Inanimate NCC Distribution', 'Small Inanimate NCC Distribution']

for i in range(4):
    group_mean = np.squeeze(np.mean(histogram_list[(60 * i):(60 * (i+1))], axis=0))
    group_sd = np.squeeze(np.std(histogram_list[(60 * i):(60 * (i+1))], axis=0))
    ax[i//2, np.mod(i, 2)].plot(bins, group_mean, color=colors[i])
    ax[i//2, np.mod(i, 2)].fill_between(bins, group_mean - group_sd, group_mean + group_sd, alpha=0.5, color=colors[i])
    ax[i//2, np.mod(i, 2)].set_xlabel(r'Normalized Contour Curvature $\hat{\kappa}$', fontsize=14)
    ax[i//2, np.mod(i, 2)].set_ylabel(r'Probability Density $P(\hat{\kappa})$', fontsize=14)
    ax[i//2, np.mod(i, 2)].set_title(titles[i], fontsize=16)

plt.tight_layout()

## ----- Calculate And Save NCC Distributions For Each Of Four Subgroups ----- ##

category_filenames = ['big_animate_mean_ncc', 'small_animate_mean_ncc', 'big_inanimate_mean_ncc', 'small_inanimate_mean_ncc']

for i in range(4):
    group_mean = np.squeeze(np.mean(histogram_list[(60 * i):(60 * (i+1))], axis=0))/dKappa
    group_sd = np.squeeze(np.std(histogram_list[(60 * i):(60 * (i+1))], axis=0))/dKappa
    #with open(stimuli_dir + '/' + category_filenames[i] + '.pickle', 'wb') as pickle_file:
    #    pickle.dump(group_mean, pickle_file)

## ----- Visualize Some Example Distributions ----- ##

categories_fig1 = ['/Big-Animate/', '/Small-Animate/', '/Big-Inanimate/', '/Small-Inanimate/']
filenames = ['0.96_hoofed-moose1.jpg', '0.90_bunny2.jpg', '0.47_telephonebooth.jpg', '1.50_Aspoolofthre6.jpg']
files = [stimuli_dir + categories_fig1[i] + filenames[i] for i in range(4)]

plt.rcParams.update({'font.size': 14})

plt.figure(figsize = (13, 10))
for i in range(4):
    image = processImage(files[i])
    kappa, kappaDist, bins, f = contourCurvature(image, rho, num_bins, bg_threshold)

    plt.subplot(4, 4, 4 * i + 1)
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.subplot(4, 4, 4 * i + 2)
    plt.imshow(f)
    plt.axis('off')
    plt.subplot(4, 4, 4 * i + 3)
    plt.imshow(kappa)
    plt.axis('off')
    plt.subplot(4, 4, 4 * i + 4)
    plt.plot(bins, histogram_list[all_images.index(files[i]),:], color='royalblue')
    plt.xlabel(r'$\hat{\kappa}$')
    plt.ylabel(r'$P(\hat{\kappa})$')

plt.tight_layout()


## ----- Visualize Some Example Distributions ----- ##

categories_fig1 = ['/Big-Animate/', '/Small-Animate/', '/Big-Inanimate/', '/Small-Inanimate/']
filenames = ['0.96_hoofed-moose1.jpg', '0.83_mouse.jpg', '0.47_telephonebooth.jpg', '1.46_computer mouse.jpg']
files = [stimuli_dir + categories_fig1[i] + filenames[i] for i in range(4)]

plt.rcParams.update({'font.size': 14})

plt.figure(figsize = (8, 12))
for i in range(4):
    image = processImage(files[i])
    kappa, kappaDist, bins, f = contourCurvature(image, rho=0.04, numBins=101, background_threshold=5)

    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.subplot(4, 2, 2 * i + 2)
    plt.plot(bins, kappaDist, color='royalblue')
    plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$', fontsize=14)
    plt.ylabel(r'Probability Density $P(\hat{\kappa})$', fontsize=14)

plt.tight_layout()


categories_fig1 = ['/Small-Animate/', '/Small-Inanimate/', '/Big-Animate/', '/Big-Inanimate/']

filenames = ['1.59_hedgehog.jpg', '1.29_ASERVICEB.jpg', '0.96_hoofed-moose1.jpg', '0.47_telephonebooth.jpg']
titles = ['Hedgehog NCC', 'Bell NCC', 'Moose NCC', 'Telephone Booth NCC']
files = [stimuli_dir + categories_fig1[i] + filenames[i] for i in range(4)]

plt.rcParams.update({'font.size': 14})

plt.figure(figsize = (13, 6))
for i in range(4):
    image = processImage(files[i])
    kappa, kappaDist, bins, f = contourCurvature(image, rho=0.04, numBins=101, background_threshold=5)
    plt.subplot(2, 4, 2 * i + 1)
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.subplot(2, 4, 2 * i + 2)
    plt.plot(bins, kappaDist, color='royalblue')
    #plt.imshow(kappa)
    plt.xlabel(r'$\hat{\kappa}$')
    plt.ylabel(r'$P(\hat{\kappa})$')
    plt.title(titles[i])

plt.tight_layout()


## ----- Visualize Processing Steps for An Example Image ----- ##

plt.rcParams.update({'font.size': 10})

## --- supplementary figure (S1) ---

#filename = stimuli_dir + '/Small-Animate/1.58_chipmunk.jpg'
#filename = stimuli_dir + '/Small-Inanimate/0.60_Acombination8.jpg'
#filename = stimuli_dir + '/Small-Inanimate/1.05_B_coffee.jpg'
filename = stimuli_dir + '/Big-Inanimate/1.59_picnictable.jpg'

image = processImage(filename)
kappa, kappaDist, bins, f, paddedImage = contourCurvature(image, rho, num_bins, bg_threshold, False, False, True)

# plot
plt.figure(figsize=(5, 6))
plt.subplot(3, 2, 1)
plt.imshow(paddedImage, cmap='gray_r')
plt.axis('off')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.imshow(f, cmap='gray_r')
plt.axis('off')
plt.title('Filtered Image')

plt.subplot(3, 2, 3)
plt.contour(f)
plt.gca().invert_yaxis()
plt.axis('off')
plt.title('Contour Plot')
plt.axis('equal')

plt.subplot(3, 2, 4)
plt.imshow(kappa)
plt.axis('off')
plt.title('NCC')
plt.colorbar()
plt.axis('equal')

plt.subplot(3, 1, 3)
plt.plot(bins, kappaDist, color='royalblue')
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$')
plt.ylabel(r'Probability Density $P(\hat{\kappa})$')
plt.title('Normalized Contour Curvature Distribution')

plt.tight_layout()

## ----- Visualize Processing Steps for An Example Image ----- ##

plt.rcParams.update({'font.size': 10})

## --- supplementary figure (S1) ---

#filename = '../AnimacySize/Small-Animate/1.58_chipmunk.jpg'
#filename = '../AnimacySize/Small-Inanimate/0.60_Acombination8.jpg'
#filename = '../AnimacySize/Small-Inanimate/1.05_B_coffee.jpg'
filename = stimuli_dir + '/Small-Animate/1.58_chipmunk.jpg'

image = processImage(filename)
kappa, kappaDist, bins, f, paddedImage = contourCurvature(image, rho, num_bins, bg_threshold, False, False, True)

# plot
plt.figure(figsize=(5, 6))
plt.subplot(3, 2, 1)
plt.imshow(paddedImage, cmap='gray_r')
plt.axis('off')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.imshow(f, cmap='gray_r')
plt.axis('off')
plt.title('Filtered Image')

plt.subplot(3, 2, 3)
plt.contour(f)
plt.gca().invert_yaxis()
plt.axis('off')
plt.title('Contour Plot')
plt.axis('equal')

plt.subplot(3, 2, 4)
plt.imshow(kappa)
plt.axis('off')
plt.title('NCC')
plt.colorbar()
plt.axis('equal')

plt.subplot(3, 1, 3)
plt.plot(bins, kappaDist, color='royalblue')
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$')
plt.ylabel(r'Probability Density $P(\hat{\kappa})$')
plt.title('Normalized Contour Curvature Distribution')

plt.tight_layout()

## ----- Visualize Processing Steps for An Example Image ----- ##

plt.rcParams.update({'font.size': 12})

# calculate for specific example
filename = stimuli_dir + '/Small-Inanimate/0.55_lightbulb.jpg'
image = processImage(filename)
kappa, kappaDist, bins, f = contourCurvature(image, rho, num_bins, bg_threshold, False, False)

# plot
plt.figure(figsize=(14, 3))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(f, cmap='gray_r')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(kappa)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.plot(bins, kappaDist, color='royalblue')
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$')
plt.ylabel(r'Probability Density $P(\hat{\kappa})$')
plt.title('NCC Distribution')
plt.tight_layout()

plt.show()
