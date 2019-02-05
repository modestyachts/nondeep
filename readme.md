# Instructions

## Setup
The requirments to run the code in this repo are:

0. A valid AWS account
1. awscli installed `pip install awscli`
2. awscli configured `aws configure`
3. pywren `pip install pywren`
4. pywren setup `pywren-setup`
5. pytorch `conda install pytorch`


## Fisher Vectors
This code is based off work at INRIA by Sanchez et Al.
https://hal.inria.fr/hal-00830491v2/document. 
It is a python port of the pipeline described in their paper.

We expect that a fishervector model trained with least squares and 256 GMM centers (65k features)
will achieve 55% top-5 accuracy.

## Random Features
This code is based off work by Coates & Ng:
https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
We remove the kmeans method in their paper and present a purely feedforward random approach.

We expect that a cifar model trained with least squares and 4096 random features (with the default convolutional parameters)
will achieve 82% top-1 accuracy




