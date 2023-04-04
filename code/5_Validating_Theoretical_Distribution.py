"""5. Validating Theoretical Distribution

This script demonstrates and numerically validates the NCC probability distributions derived
for Gaussian-correlated Gaussian random fields.
"""

import numpy as np
import cv2
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy.io
import time
import cvxpy as cp

## ---- COMPARE MONTE-CARLO WITH THEORETICAL DISTRIBUTION ---- ##

# number of random curvatures to generate
numSamples = 1000000

# construct histogram bins for curvature (kappa)
dKappa = 2
kappaMin = -50
kappaMax = 50
kappaEdges = np.arange(start = kappaMin - dKappa/2, stop = kappaMax + dKappa, step = dKappa)
kappaBins = kappaEdges[0:-1] + dKappa / 2

# construct histogram bins for NCC (kappaNorm)
num_bins = 50
kappaNormMin = 0
kappaNormMax = 1
kappaNormEdges = np.linspace(-1, 1, num_bins+1)
dKappaNorm = kappaNormEdges[1] - kappaNormEdges[0]
kappaNormBins = kappaNormEdges[0:-1] + dKappaNorm / 2

# correlation length
xi_c = 0.2

## ---- Generate Samples ---- ##

# generate uncorrelated gaussian samples (has mean=0 and std=1)
rawGaussians = np.random.normal(0, 1, size=(6, numSamples))

# factor covariance matrix
T = np.diag([1/xi_c, 1/xi_c, 1/xi_c**2, np.sqrt(3)/xi_c**2, 2*np.sqrt(2/3)/xi_c**2, 1/np.sqrt(2)])
T[3, 4] = 1/(np.sqrt(3) * xi_c**2)
T[3, 5] = - 1/np.sqrt(3)
T[4, 5] = - 1/np.sqrt(6)

# induce correlations
gaussianSamples = T.T @ rawGaussians

# choose only samples above mean
fieldValues = gaussianSamples[5, :]
derivatives = gaussianSamples[0:5, fieldValues >= 0] 

# calculate derivatives %
fx = derivatives[0,:]
fy = derivatives[1,:]
fxy = derivatives[2,:]
fxx = derivatives[3,:]
fyy = derivatives[4,:]

# calculate contour curvatures %
contourCurvatures = (- fxx * fy**2 - fyy * fx**2 + 2 * fx * fy * fxy) / ((fx**2 + fy**2)**(3/2))
normalizedCurvatures = contourCurvatures / ( 2 + np.abs(contourCurvatures) )

# calculate histogram %
kappaHist = np.histogram(np.ndarray.flatten(contourCurvatures), kappaEdges)
kappaNormHist = np.histogram(np.ndarray.flatten(normalizedCurvatures), kappaNormEdges)

# convert to distribution %
kappaDist = kappaHist[0] / np.sum(kappaHist[0][0:-1])
kappaNormDist = kappaNormHist[0] / np.sum(kappaNormHist[0][0:-1])

## ---- Theoretical Predictionn ---- ##

def curvatureCDF(k, mu, gam2):
    sigma = np.sqrt(gam2 - mu**2)
    last_term = (1/2 + 1/np.pi * np.arctan(mu/sigma * (-k) / np.sqrt(gam2 + (-k)**2)))
    return 1 - ( 1/2 - 1/np.pi * np.arctan(mu/sigma) + (-k)/np.sqrt(gam2 + (-k)**2) * last_term)

mu = - 1 / xi_c
gamma2 = 3 / xi_c**2

# calculate theoretical CDFs %
kappaCDF = curvatureCDF(kappaEdges, mu, gamma2)
kappaNormCDF = curvatureCDF(2*kappaNormEdges / ( 1 - np.abs(kappaNormEdges)), mu, gamma2)
kappaNormCDF[0] = 0
kappaNormCDF[-1] = 1

# fix normalization due to having no overflow bin for non-normalized curvature %
kappaCDF = kappaCDF - kappaCDF[0]
kappaCDF = kappaCDF / kappaCDF[-1]

## ---- plot comparison! ----

plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.bar(kappaBins, kappaDist, width = 1.5, facecolor='yellowgreen')
plt.plot(kappaBins, np.diff(kappaCDF), 'blue', linewidth=2.5)
plt.xlabel(r'Contour Curvature $\kappa$', fontsize=14)
plt.ylabel(r'Probability Density $P(\kappa)$', fontsize=14)
plt.title('Contour Curvature Distribution', fontsize=16)
plt.legend(['Theory', 'Numerics'], fontsize=14)

plt.subplot(1, 2, 2)
plt.bar(kappaNormBins, kappaNormDist, width = 0.03, facecolor='yellowgreen')
plt.plot(kappaNormBins, np.diff(kappaNormCDF), 'blue', linewidth=2.5)
plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$', fontsize=14)
plt.ylabel(r'Probability Density $P(\hat{\kappa})$', fontsize=14)
plt.title('Normalized Contour Curvature Distribution', fontsize=16)
plt.legend(['Theory', 'Numerics'], fontsize=14)

## ---- Effect of Gaussian Filtering on NCC Distribution ---- ##

def cumulativeDistributionFunction(kappa, xi):
    temp_term = 1/2 - 1/np.pi * np.arctan(((-kappa) * xi)/(np.sqrt(2) * np.sqrt(3 + (-kappa)**2 * xi**2)))
    C = 1/2 + 1/np.pi * np.arctan(1/np.sqrt(2)) + (-kappa) * xi / np.sqrt(3 + (-kappa)**2 * xi**2) * temp_term
    return 1 - C

xi0 = 0.1
sigma = [0, 0.2, 0.5, 1.0, 2.0]

plt.figure(figsize=(8, 5))
for i in range(len(sigma)):
    new_xi = np.sqrt(xi0**2 + 2 * sigma[i]**2)
    c = cumulativeDistributionFunction(2 * kappaNormBins / (1 - np.abs(kappaNormBins)), new_xi)
    p = np.diff(c)/dKappaNorm
    plt.plot(kappaNormBins[0:-1], p, linewidth=2)

plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$', fontsize=14)
plt.ylabel(r'Probability Density $P(\hat{\kappa})$', fontsize=14)
plt.title('Effect of Gaussian Filtering on NCC Distribution', fontsize=16)
plt.legend([r'$\sigma = 0.0$', r'$\sigma = 0.2$', r'$\sigma = 0.5$', r'$\sigma = 1.0$', r'$\sigma = 2.0$'], fontsize=14)

## ---- Optimizing for Optimal Correlation Length Distribution ---- ##

def solveForXiDistribution(ncc_histogram):
    # get probability distribution
    kappaProb = ncc_histogram/np.sum(ncc_histogram)

    # matrix size parameters.
    n = len(xi_vec)

    # initialize decision variable
    x = cp.Variable(shape=n)

    # define problem (cross-entropy minimization)
    obj = cp.Minimize(kappaProb * (cp.log(kappaProb.T) - cp.log(PMFs.T * x)))
    constraints = [np.ones((1, len(xi_vec))) * x == 1,
                  np.identity(n) * x <= np.ones(n),
                  np.identity(n) * x >= np.zeros(n)]
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, max_iters = 200, abstol=1e-10, reltol=1e-10)

    # get ncc probability distribution corresponding to this xi distribution
    kappaProb_fit = x.value.T @ PMFs
    return x.value, kappaProb_fit


# smoothing parameter
rho = 0.04

# number of bins used in NCC histogram
num_bins = 201

# construct histogram bins for curvature (kappa)
dKappa = 2
kappaMin = -50
kappaMax = 50
kappaEdges = np.arange(start = kappaMin - dKappa/2, stop = kappaMax + dKappa, step = dKappa)
kappaBins = kappaEdges[0:-1] + dKappa / 2

# construct histogram bins for NCC (kappaNorm)
kappaNormMin = 0
kappaNormMax = 1
kappaNormEdges = np.linspace(-1, 1, num_bins+1)
dKappaNorm = kappaNormEdges[1] - kappaNormEdges[0]
kappaNormBins = kappaNormEdges[0:-1] + dKappaNorm / 2

# prepare vectors for the correlation length and NCC values
xi_vec = np.linspace(0.005, 4, 400)
kappa = 2 * kappaNormEdges[1:-1] / (1 - np.abs(kappaNormEdges[1:-1]))

# construct grid for kappa and xi
xi_vec = np.reshape(xi_vec, (len(xi_vec), 1))
kappa = np.reshape(kappa, (1, len(kappa)))

XI = np.tile(xi_vec, (1, len(kappa)))
KAPPA = np.tile(kappa, (len(xi_vec), 1))

# calculate CDF and PMF
CDFs = np.arctan(np.sqrt(2))/np.pi - (-KAPPA) * XI * ( 1/2 - 1/np.pi * np.arctan((-KAPPA) * XI / np.sqrt(6 + 2 * (-KAPPA)**2 * XI**2) ) ) / np.sqrt(3 + (-KAPPA)**2 * XI**2)
CDFs = np.hstack((np.zeros((len(xi_vec), 1)), CDFs, np.ones((len(xi_vec), 1))))
PMFs = np.diff(CDFs, 1, 1)

# make category filenames
category_filenames = {'big_animate': 'big_animate_mean_ncc',
                      'small_animate': 'small_animate_mean_ncc',
                      'big_inanimate': 'big_inanimate_mean_ncc',
                      'small_inanimate': 'small_inanimate_mean_ncc'}

# calculations are stored here
calculation_dir = '../calculations'

# for each category, solve for distribution and plot
group_means = {}
xi_dists = {}
for cat in category_filenames.keys():
    with open(calculation_dir + '/' + category_filenames[cat] + '.pickle', 'rb') as pickle_file:
        group_mean = pickle.load(pickle_file)
        group_means[cat] = np.interp(np.linspace(-1, 1, 201), np.linspace(-1, 1, 101), group_mean)
        xi_dists[cat] = solveForXiDistribution(group_means[cat])


## ---- Plot the results ---- ##

class_names = {'big_animate': 'Large Animate', 'small_animate': 'Small Animate',
               'big_inanimate': 'Large Inanimate', 'small_inanimate': 'Small Inanimate'}

peak_names_1 = [r'$p^{\xi}_1$', r'$p^{\xi}_2$', r'$p^{\xi}_3$', r'$p^{\xi}_4$']
peak_names_2 = [r'$\Xi \ p^{\xi}_1$', r'$\Xi \ p^{\xi}_2$', r'$\Xi \ p^{\xi}_3$', r'$\Xi \ p^{\xi}_4$']

xi_dist_splits = {}
xi_dist_splits['big_animate'] = [0, 7, 25, 400]
xi_dist_splits['small_animate'] = [0, 5, 20, 400]
xi_dist_splits['big_inanimate'] = [0, 7, 20, 70, 400]
xi_dist_splits['small_inanimate'] = [0, 7, 20, 50, 400]

i = 0
plt.figure(figsize=(13, 16))
for cat in category_filenames.keys():
    plt.subplot(4, 2, 2 * i + 1)
    plt.title(class_names[cat], fontsize=18)
    plt.xlabel(r'Scaled Correlation Length $\xi$', fontsize=16)
    plt.ylabel(r'Probability Density $P(\xi)$', fontsize=16)
    plt.plot(xi_vec, np.zeros(xi_dists[cat][0].shape), color='k')

    plt.subplot(4, 2, 2 * i + 2)
    plt.title(class_names[cat], fontsize=18)
    plt.ylabel(r'Probability Density $P(\hat{\kappa})$', fontsize=16)
    plt.xlabel(r'Normalized Contour Curvature $\hat{\kappa}$', fontsize=16)
    plt.plot(kappaNormEdges[1:], group_means[cat]/(dKappaNorm * np.sum(group_means[cat])), 'k--', label='Data')
    plt.plot(kappaNormEdges[1:], xi_dists[cat][1]/(dKappaNorm * np.sum(xi_dists[cat][1])), 'k', label='Model Fit')

    for j in range(len(xi_dist_splits[cat]) - 1):
      indicator = np.zeros(xi_dists[cat][0].shape)
      indicator[xi_dist_splits[cat][j]:xi_dist_splits[cat][j+1]] = np.ones(xi_dist_splits[cat][j+1] - xi_dist_splits[cat][j])
      partial_xi_dist = xi_dists[cat][0] * indicator
      plt.subplot(4, 2, 2*i+1)
      plt.plot(xi_vec, partial_xi_dist/(xi_vec[1] - xi_vec[0]), label=peak_names_1[j])
      plt.subplot(4, 2, 2*i+2)
      kappaProb_fit = partial_xi_dist.T @ PMFs
      plt.plot(kappaNormEdges[1:], kappaProb_fit.T/dKappaNorm)

    plt.subplot(4, 2, 2 * i + 1)
    plt.legend(loc='upper right', fontsize=14)
    
    plt.subplot(4, 2, 2 * i + 2)
    plt.legend(loc='upper left', fontsize=14)
    i += 1

plt.tight_layout()
plt.show()
