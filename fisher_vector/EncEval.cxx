 /** @internal
 ** @file     FisherExtractor.cxx
 ** @brief    JNI Wrapper for enceval GMM and Fisher Vector
 **/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include <gmm.h>
#include <fisher.h>

extern "C" {
  float* ctypes_get_fisher_features(float* means, int means_length, int n_dim, int n_gauss, float* covariances, int covariances_length, float* priors, int priors_length, float* descs, int descs_length);
  void ctypes_compute_gmm(int n_gaus, int n_dim, float* gmm_samples, int num_samples, int seed, float* means, float* vars, float* coefs);
}

float* ctypes_get_fisher_features(float* means, int means_length, int n_dim, int n_gauss, float* covariances, int covariances_length, float* priors, int priors_length, float* descs, int descs_length) {
  float * fk = 0;
  // malloc the result
  uint32_t fvenc_length = 2 * n_dim * n_gauss;
  fk = (float*) malloc( sizeof(float) * fvenc_length);
  if ( fk == NULL ) {
    printf("Error allocating memory for the FVenc buffer\n");
    fflush(stdout);
    exit(-1);
  }
  printf("input to vectors\n");
  fflush(stdout);
  std::vector<float*> mean(n_gauss);
  std::vector<float*> variance(n_gauss);
  for (int j = 0; j < n_gauss; ++j) {
    mean[j] = &means[j*n_dim];
    variance[j] = &covariances[j*n_dim];
  }
  std::vector<float> coef(priors, priors + n_gauss);
  // prepare a GMM model with data from the structure
  gaussian_mixture<float> gmmproc(n_gauss,n_dim);
  printf("make gmm\n");
  fflush(stdout);
  gmmproc.set(mean, variance, coef);
  // construct a c++ struct with default parameter values
  // in the Mex the settings are sent as a third variable, we stick to defaults.
  fisher_param fisher_encoder_params;
  fisher_encoder_params.alpha = 1.0f;
  fisher_encoder_params.pnorm = 0.0f;
  fisher_encoder_params.print();
  // What is the role of this fhisher_handle.. can we perhaps keep it between calls.. (save time)?
  printf("make handle \n");
  fflush(stdout);
  fisher<float> fisher_encoder(fisher_encoder_params);
  // initialise encoder with a GMM model (vocabulary)
  printf(".. and set gmm model to handle \n");
  fflush(stdout);
  fisher_encoder.set_model(gmmproc);

  printf("descriptors to vector \n");
  printf("descriptors length: %d\n", descs_length);
  fflush(stdout);
  std::vector<float*> x(descs_length);

  for (int j = 0; j < descs_length; ++j) {
    x[j] = &descs[j*n_dim];
  }
  bool weights = false;
  // load in weights if specified
  // do encoding
  printf("encode without weights \n");
  fflush(stdout);
  fisher_encoder.compute(x, fk);
  return fk;
}

void ctypes_compute_gmm(int n_gauss, int n_dim, float* gmm_samples, int num_samples, int seed, float* means, float* vars, float* coefs) {

  //Copy to C vectors. We assume things come at us sample at a time.
  std::vector<float*> samples(num_samples);

  for (int i = 0; i < num_samples; ++i) {
    samples[i] = &gmm_samples[i*n_dim];
  }

  printf("Samples[998][61] %f\n", samples[998][61]);
  //Create a default empty gmm parameter set.
  em_param gmm_params;
  gaussian_mixture<float> gmmproc(gmm_params, n_gauss, n_dim);

  //We don't yet accept initial means/variances/coefs yet.
  //int seed = time(NULL);
  gmmproc.random_init(samples, seed);

  //Run EM.
  gmmproc.em(samples);

  //Copy final stuff back out.
  int meanResSize = n_gauss*n_dim;
  int varResSize = n_gauss*n_dim;
  int coefResSize = n_gauss;
  int totalResSize = meanResSize + varResSize + coefResSize;

  for (int j = 0; j < n_gauss; ++j) {
    float* componentmean = gmmproc.get_mean(j);
    float* componentvariance = gmmproc.get_variance(j);

    for (int i = 0; i < n_dim; ++i) {
      means[i+j*n_dim] = componentmean[i];
      vars[i+j*n_dim] = componentvariance[i];
    }
    coefs[j] = gmmproc.get_mixing_coefficients(j);
  }
}
