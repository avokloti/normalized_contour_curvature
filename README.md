# normalized_contour_curvature

This repository contains code associated with "Cognitive Categorization Using Contour Curvature" (A. Marantan, I. Tolkova, L. Mahadevan). It is divided across six parts:

1. Examples and Interpretation: this script shows the application and interpretation of normalized contour curvature (NCC) when applied to a dataset of object images, varying in animacy and size.

2. Analysis of Animacy/Size Stimuli: this script implements a Bayesian binary classifier to distinguish between the animacy (animate vs inanimate) and size (large vs small) object categories based solely on NCC.

3. Analysis of Text/Face/Tetris Stimuli: this script characterizes a dataset of alpha-numeric characters, cartoon faces, and Tetris shapes through principal component analysis over normalized contour curvature.

4. Analysis of Viewpoint and Illumination: this script examines robustness across varying viewpoint and illumination conditions by examining the images from the Amsterdam Library of Object Images with multidimensional scaling.

5. Validating Theoretical Model: this script numerically validates the NCC probability distributions derived for Gaussian-correlated Gaussian random fields.

6. Generative Model: this script presents implements the model for generating artificial images for a given NCC distribution.
