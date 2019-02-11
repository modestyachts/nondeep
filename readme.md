# Instructions

Hello this is a repository to host two non-deep image classification models and their training code.


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

With 256 GMM centers an ImageNet trained model should get to 55% top-5 accuracy (35.1  top-1 accuracy)

To train + eval the model
1. Generate features from stored S3 keys usinn PyWren 
   `` python featurize_fisher_model.py --num_centers 256 ```
   (if it errors run the following command to *resume*)
   ```  python featurize_fisher_model.py --num_centers 256 --use_cache_gmm_sift --use_cache_gmm_lcs ```
 2. Train model using least squares (file fv_model contains the weights of the model)
  ``` python train_fisher_model.py fishervector_features.pickle fv_model ```
   

## Random Features
This code is based off work by Coates & Ng:
https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
We remove the kmeans method in their paper and present a purely feedforward random approach.

With 256k random convolutional filters a Cifar-10 trained model should get to 85.6 top-1 accuracy.

To train + eval the model
1. Just run the script
  ``` python featurize_train_model.py ```
  




