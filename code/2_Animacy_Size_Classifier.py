"""2. Classification of Size and Animacy

This script implements a Bayesian binary classifier to distinguish between
the animacy (animate vs inanimate) and size (large vs small) object categories
using the normalized contour curvature (NCC) descriptor.
"""

import numpy as np
import cv2
import os
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# import functions
from functions import *

## ----- Prepare Data Paths and Labels, and Calculate Histograms ----- ##

# flag to indicate whether data should be calculated or loaded
CALCULATE = False

# set outer working directory
stimuli_dir = '../Stimuli/AnimacySize'

# folder names
categories = ['/Big-Animate/', '/Big-Inanimate/', '/Small-Animate/', '/Small-Inanimate/']

# collect all filenames
big_animate = os.listdir(stimuli_dir + categories[0])
big_inanimate = os.listdir(stimuli_dir + categories[1])
small_animate = os.listdir(stimuli_dir + categories[2])
small_inanimate = os.listdir(stimuli_dir + categories[3])

# add outer directory to all filenames
big_animate = [stimuli_dir + categories[0] + path for path in big_animate]
big_inanimate = [stimuli_dir + categories[1] + path for path in big_inanimate]
small_animate = [stimuli_dir + categories[2] + path for path in small_animate]
small_inanimate = [stimuli_dir + categories[3] + path for path in small_inanimate]

# take all images
all_images = big_animate + small_animate + big_inanimate + small_inanimate
num_images = len(all_images)

# set labels to differentite either animate/inanimate or big/small
labels_animacy = [1] * int(num_images/2) + [0] * int(num_images/2)
labels_size = [1] * int(num_images/4) + [0] * int(num_images/4) + [1] * int(num_images/4) + [0] * int(num_images/4)

# list of "rho" smoothing parameters to iterate through
rho_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
num_bins = 101

if (CALCULATE):
    start_time = time.time()

    # iterate through all smoothing parameters
    for rho in rho_list:
        print('Starting rho = ' + str(rho))

        # prepare a matrix to store NCC histograms
        histogram_list = np.zeros((num_images, num_bins))

        for i in range(num_images):
            # read in image and convert to grayscale float
            image = processImage(all_images[i])

            # calculate NCC
            kappa, kappaDist, bins, f = contourCurvature(image, rho, num_bins)

            # put in histogram_list
            histogram_list[i,:] = kappaDist

            # print
            if (i % 10 == 0):
                print('--- Finished calculation for image %d/%d' % (i, num_images))
            
            # save histogram by pickling
            pickle_filename = '../calculations/histogram_rho_' + str(rho)
            with open(pickle_filename, 'wb') as pickle_file:
                pickle.dump(histogram_list, pickle_file)
    
    end_time = time.time()
    print('Calculation took ' + str(end_time - start_time))

## ----- Logistic Classifier Class and Methods  ----- ##

class logisticClassifier():
    # construct logistic classifier
    def __init__(self, histograms, bins, labels, training_ratio):
        self.histograms = histograms
        self.bins = bins
        self.labels = labels
        self.num_images = histograms.shape[0]
        self.training_size = int(round(self.num_images * training_ratio))
        self.testing_size = self.num_images - self.training_size
    
    # calculate log-likelihood to classify point
    def classifyPoint(self, test_hist, class1_histogram, class2_histogram):
        log_ratio = np.log(class1_histogram[0,1:-1]/class2_histogram[0,1:-1])
        log_likelihood = np.sum(test_hist[1:-1] * log_ratio)
        return 1 * (log_likelihood > 0)

    # randomly divide into training and testing data
    def divideIntoTrainingAndTesting(self):
        ordering = np.random.permutation(np.arange(self.num_images))
        training_indices = np.sort(ordering[0:self.training_size])
        testing_indices = np.sort(ordering[self.training_size:])
        training_labels = [self.labels[i] for i in training_indices]
        testing_labels = [self.labels[i] for i in testing_indices]
        training_data = self.histograms[training_indices,:]
        testing_data = self.histograms[testing_indices,:]
        return [training_data, testing_data, training_labels, testing_labels, training_indices, testing_indices]

    # calculate mean and standard deviation of a training dataset (with optional plotting)
    def calculateMeanHistogramsForBothClasses(self, training_data, training_labels, plot_flag):
        class1_histogram = np.mean(training_data[np.where(np.asarray(training_labels)), :], axis=1)
        class2_histogram = np.mean(training_data[np.where(np.ones(self.training_size) - np.asarray(training_labels)), :], axis=1)
        class1_sd = np.std(training_data[np.where(np.asarray(training_labels)), :], axis=1)
        class2_sd = np.std(training_data[np.where(np.ones(self.training_size) - np.asarray(training_labels)), :], axis=1)
        if (plot_flag):
            fig, ax = plt.subplots(1)
            fig.set_size_inches(8, 5)
            ax.plot(self.bins, np.squeeze(class1_histogram), label='Animate')
            ax.fill_between(self.bins, np.squeeze(class1_histogram) - np.squeeze(class1_sd), np.squeeze(class1_histogram) + np.squeeze(class1_sd), alpha=0.5)
            ax.plot(self.bins, np.squeeze(class2_histogram), label='Inanimate')
            ax.fill_between(self.bins, np.squeeze(class2_histogram) - np.squeeze(class2_sd), np.squeeze(class2_histogram) + np.squeeze(class2_sd), alpha=0.5)
            plt.xlabel('NCC Value')
            plt.ylabel('Probability Density')
            plt.title('Training Data NCC Distribution')
            plt.legend()
        
        return [class1_histogram, class2_histogram, class1_sd, class2_sd]

    # run a single classification trial
    def runSingleTrial(self):
        # divide into training and testing with new ordering
        [training_data, testing_data, training_labels, testing_labels, training_indices, testing_indices] = self.divideIntoTrainingAndTesting()

        # calculate histograms for both classes from training data
        [class1_histogram, class2_histogram, class1_sd, class2_sd] = self.calculateMeanHistogramsForBothClasses(training_data, training_labels, False)

        # classify all points in testing data
        predictions = np.zeros((self.testing_size))
        for i in range(self.testing_size):
            predictions[i] = self.classifyPoint(testing_data[i,:], class1_histogram, class2_histogram)
        
        # return fraction correct
        tn, fp, fn, tp = confusion_matrix(testing_labels, predictions).ravel()
        return (tn + tp)/(tn + tp + fn + fp), tp/(tp + fn), fp/(tp + fn), class1_histogram, class2_histogram

    # run multiple classification trials, with randomized training/testing data
    def runMultipleTrials(self, num_trials):
        success = np.zeros(num_trials)
        true_pos = np.zeros(num_trials)
        false_pos = np.zeros(num_trials)
        class1_means = np.zeros((num_trials, num_bins))
        class2_means = np.zeros((num_trials, num_bins))
        for i in range(num_trials):
            success[i], true_pos[i], false_pos[i], class1_means[i,:], class2_means[i,:] = self.runSingleTrial()
        
        return success, true_pos, false_pos, class1_means, class2_means

## ----- Run Logistic Classification  ----- ##

# make histogram bins
num_bins = 101
dKappa = 2/num_bins
kappaMin = -1
kappaMax = 1
binEdges = np.linspace(kappaMin, kappaMax, num_bins+1)
bins = binEdges[0:-1] + dKappa/2

# prepare lists to save histogram means
big_histogram_list = []
small_histogram_list = []
animate_histogram_list = []
inanimate_histogram_list = []

# prepare lists to save histogram standard deviations
big_sd_list = []
small_sd_list = []
animate_sd_list = []
inanimate_sd_list = []

# prepare lists to save true positive, false positive, and success rate for size classification
true_pos_size_list = []
false_pos_size_list = []
success_size_list = []

# prepare lists to save true positive, false positive, and success rate for animacy classification
true_pos_animacy_list = []
false_pos_animacy_list = []
success_animacy_list = []

# track rho and TF values
current_rho_list = []
current_training_size_list = []

# construct lists for the "rho" values and the training fraction (TF) to sweep
rho_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
training_size_list = [0.3, 0.5, 0.7]

for rho in rho_list:
    print('Rho = ' + str(rho))
    # read in this dataset
    pickle_filename = '../calculations/histogram_rho_' + str(rho)
    with open(pickle_filename, 'rb') as pickle_file:
        histogram_list = pickle.load(pickle_file)
    
    for training_size in training_size_list:
        print('--- TF = ' + str(training_size))
        # run classifier
        classifier_size = logisticClassifier(histogram_list, bins, labels_size, training_size)
        classifier_animacy = logisticClassifier(histogram_list, bins, labels_animacy, training_size)

        # get results of 1000-fold cross-validation
        success_size, true_pos_size, false_pos_size, big_means, small_means = classifier_size.runMultipleTrials(1000)
        success_animacy, true_pos_animacy, false_pos_animacy, animate_means, inanimate_means = classifier_animacy.runMultipleTrials(1000)

        true_pos_size_list.append(true_pos_size)
        false_pos_size_list.append(false_pos_size)
        success_size_list.append(success_size)
        true_pos_animacy_list.append(true_pos_animacy)
        false_pos_animacy_list.append(false_pos_animacy)
        success_animacy_list.append(success_animacy)

        # save mean and standard deviation of class histograms over cross-validation trials
        big_histogram_list.append(np.mean(big_means, axis=0))
        small_histogram_list.append(np.mean(small_means, axis=0))
        big_sd_list.append(np.std(big_means, axis=0))
        small_sd_list.append(np.std(small_means, axis=0))

        animate_histogram_list.append(np.mean(animate_means, axis=0))
        inanimate_histogram_list.append(np.mean(inanimate_means, axis=0))
        animate_sd_list.append(np.std(animate_means, axis=0))
        inanimate_sd_list.append(np.std(inanimate_means, axis=0))

        current_rho_list.append(rho)
        current_training_size_list.append(training_size)


## ----- Plot Success Rate as a Function of Rho, Across Training Fraction  ----- ##

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(2, 3)
fig.set_size_inches(11, 6)

for i in range(len(training_size_list)):
    success_size_mean = np.zeros(10)
    success_size_sd = np.zeros(10)
    success_animacy_mean = np.zeros(10)
    success_animacy_sd = np.zeros(10)

    for j in range(len(rho_list)):
        index = len(training_size_list) * j + i
        assert(current_rho_list[index] == rho_list[j])
        assert(current_training_size_list[index] == training_size_list[i])
        success_size_mean[j] = np.mean(success_size_list[index])
        success_size_sd[j] = np.std(success_size_list[index])
        success_animacy_mean[j] = np.mean(success_animacy_list[index])
        success_animacy_sd[j] = np.std(success_animacy_list[index])
    
    ax[0, i].plot(rho_list, np.squeeze(success_animacy_mean), label='Animate', color='C2')
    ax[0, i].fill_between(rho_list, np.squeeze(success_animacy_mean) - np.squeeze(success_animacy_sd), np.squeeze(success_animacy_mean) + np.squeeze(success_animacy_sd), alpha=0.5, color='C2')
    ax[0, i].set_xlabel(r'Relative Filter Size $\rho$')
    if (i == 0):
      ax[0, i].set_ylabel('Success Rate')
    
    ax[0, i].set_title('Classifying Animacy, TF = ' + str(training_size_list[i]))
    ax[0, i].set_ylim([0.5, 1])

    ax[1, i].plot(rho_list, np.squeeze(success_size_mean), label='Large', color='C0')
    ax[1, i].fill_between(rho_list, np.squeeze(success_size_mean) - np.squeeze(success_size_sd), np.squeeze(success_size_mean) + np.squeeze(success_size_sd), alpha=0.5, color='C0')
    ax[1, i].set_xlabel(r'Relative Filter Size $\rho$')
    if (i == 0):
      ax[1, i].set_ylabel('Success Rate')
    
    ax[1, i].set_title('Classifying Size, TF = ' + str(training_size_list[i]))
    ax[1, i].set_ylim([0.5, 1])

plt.tight_layout()

## ----- Plot Success Rate as a Function of Rho, TF = 0.3  ----- ##

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(8, 4)

success_size_mean = np.zeros(10)
success_size_sd = np.zeros(10)
success_animacy_mean = np.zeros(10)
success_animacy_sd = np.zeros(10)

for j in range(len(rho_list)):
    index = len(training_size_list) * j
    assert(current_rho_list[index] == rho_list[j])
    assert(current_training_size_list[index] == 0.3)
    success_size_mean[j] = np.mean(success_size_list[index])
    success_size_sd[j] = np.std(success_size_list[index])
    success_animacy_mean[j] = np.mean(success_animacy_list[index])
    success_animacy_sd[j] = np.std(success_animacy_list[index])

ax[0].plot(rho_list, np.squeeze(success_animacy_mean), label='Animate', color='C2')
ax[0].fill_between(rho_list, np.squeeze(success_animacy_mean) - np.squeeze(success_animacy_sd), np.squeeze(success_animacy_mean) + np.squeeze(success_animacy_sd), alpha=0.5, color='C2')
ax[0].set_xlabel(r'Relative Filter Size $\rho$')
ax[0].set_ylabel('Success Rate')
ax[0].set_title('Classifying Animacy, TF = ' + str(training_size_list[0]))
ax[0].set_ylim([0.5, 1])

ax[1].plot(rho_list, np.squeeze(success_size_mean), label='Large', color='C0')
ax[1].fill_between(rho_list, np.squeeze(success_size_mean) - np.squeeze(success_size_sd), np.squeeze(success_size_mean) + np.squeeze(success_size_sd), alpha=0.5, color='C0')
ax[1].set_xlabel(r'Relative Filter Size $\rho$')
ax[1].set_ylabel('Success Rate')
ax[1].set_title('Classifying Size, TF = ' + str(training_size_list[0]))
ax[1].set_ylim([0.5, 1])

plt.tight_layout()

## ----- Plot True Positive vs False Positive Rate, Across Training Fraction ----- ##

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(2, 3)
fig.set_size_inches(13, 7)

for i in range(len(training_size_list)):
    index = 3 * len(training_size_list) + i
    assert(current_rho_list[index] == 0.04)
    assert(current_training_size_list[index] == training_size_list[i])

    im = ax[0, i].hexbin(false_pos_animacy_list[index], true_pos_animacy_list[index], gridsize=20, mincnt=0, extent=[0, 1, 0, 1], vmin = 0, vmax = 100)
    plt.colorbar(im, ax=ax[0, i], orientation='vertical')
    ax[0, i].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
    ax[0, i].set_xlim([0, 1])
    ax[0, i].set_ylim([0, 1])
    ax[0, i].set_title('Classifying Animacy, TF = ' + str(training_size_list[i]))
    ax[0, i].set_xlabel('False "Animate" Frequency')
    ax[0, i].set_aspect('equal', 'box')
    if (i == 0):
      ax[0, i].set_ylabel('True "Animate" Frequency')
    
    im = ax[1, i].hexbin(false_pos_size_list[index], true_pos_size_list[index], gridsize=20, mincnt=0, extent=[0, 1, 0, 1], vmin = 0, vmax = 100)
    plt.colorbar(im, ax=ax[1, i], orientation='vertical')
    ax[1, i].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
    ax[1, i].set_xlim([0, 1])
    ax[1, i].set_ylim([0, 1])
    ax[1, i].set_title('Classifying Size, TF = ' + str(training_size_list[i]))
    ax[1, i].set_xlabel('False "Large" Frequency')
    ax[1, i].set_aspect('equal', 'box')
    if (i == 0):
      ax[1, i].set_ylabel('True "Large" Frequency')

plt.tight_layout()

## ----- Plot True Positive vs False Positive Rate, Across TF and Rho (Animacy) ----- ##

plot_rhos = [0.02, 0.06, 0.08, 0.1]

plt.rcParams.update({'font.size': 12})

# now plot!
fig, ax = plt.subplots(4, 3)
fig.set_size_inches(12,16)

for i in range(len(training_size_list)):
    for j in range(len(plot_rhos)):
        index = int(plot_rhos[j] * 100 - 1) * len(training_size_list) + i
        assert(current_rho_list[index] == plot_rhos[j])
        assert(current_training_size_list[index] == training_size_list[i])

        im = ax[j, i].hexbin(false_pos_animacy_list[index], true_pos_animacy_list[index], gridsize=20, mincnt=0, extent=[0, 1, 0, 1], vmin = 0, vmax = 100)
        ax[j, i].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
        ax[j, i].set_xlim([0, 1])
        ax[j, i].set_ylim([0, 1])
        ax[j, i].set_title(r'Animacy: $\rho$ = ' + str(plot_rhos[j]) + ', TF = ' + str(training_size_list[i]))
        ax[j, i].set_xlabel('False "Animate" Frequency')
        #ax[0, i].set_ylabel('True "Animate" Frequency')
        ax[j, i].set_aspect('equal', 'box')
        if (i == 0):
          ax[j, i].set_ylabel('True "Animate" Frequency')

plt.tight_layout()

## ----- Plot True Positive vs False Positive Rate, Across TF and Rho (Size) ----- ##

fig, ax = plt.subplots(4, 3)
fig.set_size_inches(12,16)

for i in range(len(training_size_list)):
    for j in range(len(plot_rhos)):
        index = int(plot_rhos[j] * 100 - 1) * len(training_size_list) + i
        assert(current_rho_list[index] == plot_rhos[j])
        assert(current_training_size_list[index] == training_size_list[i])

        im = ax[j, i].hexbin(false_pos_size_list[index], true_pos_size_list[index], gridsize=20, mincnt=0, extent=[0, 1, 0, 1], vmin = 0, vmax = 100)
        ax[j, i].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
        ax[j, i].set_xlim([0, 1])
        ax[j, i].set_ylim([0, 1])
        ax[j, i].set_title(r'Size: $\rho$ = ' + str(plot_rhos[j]) + ', TF = ' + str(training_size_list[i]))
        ax[j, i].set_xlabel('False "Large" Frequency')
        ax[j, i].set_aspect('equal', 'box')
        if (i == 0):
          ax[j, i].set_ylabel('True "Large" Frequency')

plt.tight_layout()

## ----- Classification Results for Rho = 0.04 and TF = 0.3 ----- ##

pickle_filename = '../calculations/histogram_rho_' + str(0.04)
with open(pickle_filename, 'rb') as pickle_file:
    histogram_list = pickle.load(pickle_file)

# run classifier for both big/small and animate/inanimate
classifier_size = logisticClassifier(histogram_list, bins, labels_size, 0.3)
classifier_animacy = logisticClassifier(histogram_list, bins, labels_animacy, 0.3)

# get results of 1000-fold cross-validation
success_size, true_pos_size, false_pos_size, big_means, small_means = classifier_size.runMultipleTrials(1000)
success_animacy, true_pos_animacy, false_pos_animacy, animate_means, inanimate_means = classifier_animacy.runMultipleTrials(1000)

# calculate mean and standard deviation of class histograms over cross-validation trials
big_histogram = np.mean(big_means, axis=0)
small_histogram = np.mean(small_means, axis=0)
big_sd = np.std(big_means, axis=0)
small_sd = np.std(small_means, axis=0)

animate_histogram = np.mean(animate_means, axis=0)
inanimate_histogram = np.mean(inanimate_means, axis=0)
animate_sd = np.std(animate_means, axis=0)
inanimate_sd = np.std(inanimate_means, axis=0)

# print results
print('True Positive Size: %.2f +/- %.2f' % (np.mean(true_pos_size), np.std(true_pos_size)))
print('False Positive Size: %.2f +/- %.2f' % (np.mean(false_pos_size), np.std(false_pos_size)))
print('True Positive Animacy: %.2f +/- %.2f' % (np.mean(true_pos_animacy), np.std(true_pos_animacy)))
print('False Positive Animacy: %.2f +/- %.2f' % (np.mean(false_pos_animacy), np.std(false_pos_animacy)))

## ----- True/False Positive Visualization for Rho = 0.04 and TF = 0.3 ----- ##

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4)

im = ax[0].hexbin(false_pos_animacy, true_pos_animacy, gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
plt.colorbar(im, ax=ax[0], orientation='vertical')
ax[0].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].set_title('Classifying Animacy')
ax[0].set_xlabel('False "Animate" Frequency')
ax[0].set_ylabel('True "Animate" Frequency')
ax[0].set_aspect('equal', 'box')

im = ax[1].hexbin(false_pos_size, true_pos_size, gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
plt.colorbar(im,  ax=ax[1], orientation='vertical')
ax[1].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].set_title('Classifying Size')
ax[1].set_xlabel('False "Large" Frequency')
ax[1].set_ylabel('True "Large" Frequency')
ax[1].set_aspect('equal', 'box')

plt.tight_layout()

## ----- True/False Positive Visualization for Rho = 0.04 and TF = 0.3 ----- ##


plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 4)

im = ax[0].hexbin(false_pos_animacy, true_pos_animacy, gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
plt.colorbar(im, ax=ax[0], orientation='vertical')
ax[0].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].set_title('Classifying Animacy')
ax[0].set_xlabel('False "Animate" Frequency')
ax[0].set_ylabel('True "Animate" Frequency')
ax[0].set_aspect('equal', 'box')

im = ax[1].hexbin(false_pos_size, true_pos_size, gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
plt.colorbar(im,  ax=ax[1], orientation='vertical')
ax[1].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].set_title('Classifying Size')
ax[1].set_xlabel('False "Large" Frequency')
ax[1].set_ylabel('True "Large" Frequency')
ax[1].set_aspect('equal', 'box')

plt.tight_layout()

## ----- Run Logistic Classification on Group-wise level ----- ##

# only give a portion of the histograms and labels for training!
rho = 0.04
training_size = 0.3

# make histogram bins
num_bins = 101
dKappa = 2/num_bins
kappaMin = -1
kappaMax = 1
binEdges = np.linspace(kappaMin, kappaMax, num_bins+1)
bins = binEdges[0:-1] + dKappa/2

# read in this dataset
pickle_filename = '../calculations/histogram_rho_' + str(rho)
with open(pickle_filename, 'rb') as pickle_file:
    histogram_list = pickle.load(pickle_file)

## -------------
labels_animacy = np.array(labels_animacy)
labels_size = np.array(labels_size)

true_pos_list = []
false_pos_list = []

indices = np.concatenate((range(0, 60), range(120, 180)))
classifier = logisticClassifier(histogram_list[indices], bins, labels_animacy[indices], training_size)
success, true_pos, false_pos, group1_means, group2_means = classifier.runMultipleTrials(1000)
true_pos_list.append(true_pos)
false_pos_list.append(false_pos)

indices = np.concatenate((range(60, 120), range(180, 240)))
classifier = logisticClassifier(histogram_list[indices], bins, labels_animacy[indices], training_size)
success, true_pos, false_pos, group1_means, group2_means = classifier.runMultipleTrials(1000)
true_pos_list.append(true_pos)
false_pos_list.append(false_pos)

indices = range(0, 120)
classifier = logisticClassifier(histogram_list[indices], bins, labels_size[indices], training_size)
success, true_pos, false_pos, group1_means, group2_means = classifier.runMultipleTrials(1000)
true_pos_list.append(true_pos)
false_pos_list.append(false_pos)

indices = range(120, 240)
classifier = logisticClassifier(histogram_list[indices], bins, labels_size[indices], training_size)
success, true_pos, false_pos, group1_means, group2_means = classifier.runMultipleTrials(1000)
true_pos_list.append(true_pos)
false_pos_list.append(false_pos)

fig, ax = plt.subplots(2, 2)
fig.set_size_inches(10, 9)

im = ax[0, 0].hexbin(false_pos_list[0], true_pos_list[0], gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
#plt.colorbar(im, ax=ax[0, 0], orientation='vertical')
ax[0, 0].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[0, 0].set_xlim([0, 1])
ax[0, 0].set_ylim([0, 1])
ax[0, 0].set_title('Classifying Large Objects by Animacy', fontsize=16)
ax[0, 0].set_xlabel('False "Animate" Frequency', fontsize=14)
ax[0, 0].set_ylabel('True "Animate" Frequency', fontsize=14)
ax[0, 0].set_aspect('equal', 'box')

im = ax[0, 1].hexbin(false_pos_list[1], true_pos_list[1], gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
#plt.colorbar(im, ax=ax[0, 1], orientation='vertical')
ax[0, 1].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[0, 1].set_xlim([0, 1])
ax[0, 1].set_ylim([0, 1])
ax[0, 1].set_title('Classifying Small Objects by Animacy', fontsize=16)
ax[0, 1].set_xlabel('False "Animate" Frequency', fontsize=14)
ax[0, 1].set_ylabel('True "Animate" Frequency', fontsize=14)
ax[0, 1].set_aspect('equal', 'box')


im = ax[1, 0].hexbin(false_pos_list[2], true_pos_list[2], gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
#plt.colorbar(im, ax=ax[1, 0], orientation='vertical')
ax[1, 0].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[1, 0].set_xlim([0, 1])
ax[1, 0].set_ylim([0, 1])
ax[1, 0].set_title('Classifying Animate Objects by Size', fontsize=16)
ax[1, 0].set_xlabel('False "Large" Frequency', fontsize=14)
ax[1, 0].set_ylabel('True "Large" Frequency', fontsize=14)
ax[1, 0].set_aspect('equal', 'box')

im = ax[1, 1].hexbin(false_pos_list[3], true_pos_list[3], gridsize=20, mincnt=0, extent=[0, 1, 0, 1])
#plt.colorbar(im, ax=ax[1, 1], orientation='vertical')
ax[1, 1].plot([-0.03, 1.03], [-0.03, 1.03], color='w', linewidth=3)
ax[1, 1].set_xlim([0, 1])
ax[1, 1].set_ylim([0, 1])
ax[1, 1].set_title('Classifying Inanimate Objects by Size', fontsize=16)
ax[1, 1].set_xlabel('False "Large" Frequency', fontsize=14)
ax[1, 1].set_ylabel('True "Large" Frequency', fontsize=14)
ax[1, 1].set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
