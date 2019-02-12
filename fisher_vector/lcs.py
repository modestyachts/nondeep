import numpy as np
import scipy.signal
import scipy.misc
from numpy import r_
import time
import scipy.ndimage.filters as filters



def lcs(img, stride=4, patch_size=24, pixel_border=16):
	if (len(img.shape) < 3):
		img = np.stack((img,img,img), axis=2)

	img = img.astype('float64')
	n_rows, n_cols, _ = img.shape
	(keypoint_location_x,keypoint_location_y) = np.mgrid[pixel_border+1:n_rows-pixel_border+1:stride, pixel_border+1:n_cols - pixel_border + 1:stride]
	keypoint_location = np.vstack((keypoint_location_x.ravel(), keypoint_location_y.ravel())).T
	num_key_points = keypoint_location.shape[0]
	npool_row = int(np.ceil((n_rows-2*pixel_border)/stride))
	npool_col = int(np.ceil((n_cols-2*pixel_border)/stride))
	base_points = keypoint_location[:, 0] + n_rows*(keypoint_location[:,1]-1)
	subpatchsize = int(patch_size/4)
	onesvec =  (np.ones((subpatchsize, subpatchsize))/(subpatchsize*subpatchsize))
	onesvec_1d =  (np.ones((subpatchsize))/(subpatchsize))
	mean_images = []
	std_images = []
	img_sq = (img * img).astype('float64')
	for c in range(3):
	    im_pad = np.pad(img[:, :, c], [(subpatchsize, subpatchsize), (subpatchsize, subpatchsize)], 'constant').astype('float64')
	    conv = filters.convolve1d(im_pad, onesvec_1d, axis=0)
	    conv = filters.convolve1d(conv, onesvec_1d, axis=1)[subpatchsize:-subpatchsize, subpatchsize:-subpatchsize]
	    print("Convolutions")
	    mean_images.append(conv)

	    #sq = scipy.signal.convolve2d(im_pad * im_pad, onesvec, boundary='wrap')[subpatchsize:-subpatchsize, subpatchsize:-subpatchsize]
	    sq = filters.convolve1d(im_pad * im_pad, onesvec_1d, axis=0)
	    sq = filters.convolve1d(sq, onesvec_1d, axis=1)[subpatchsize:-subpatchsize, subpatchsize:-subpatchsize]
	    sd = np.sqrt(np.maximum(sq - conv * conv, 0))
	    std_images.append(sd)
	(lcs_x_displacement,lcs_y_displacement, chan_idx)= np.meshgrid(r_[np.arange(-2,2)*subpatchsize + subpatchsize/2 -1], r_[np.arange(-2,2)*subpatchsize+subpatchsize/2 - 1], r_[np.arange(3)])
	lcs_displacement_index = lcs_x_displacement.ravel(order='F')+n_rows*lcs_y_displacement.ravel(order='F') + n_rows*n_cols*(chan_idx.ravel(order='F'))
	mean_image = np.stack(mean_images, axis=2)
	std_image = np.stack(std_images, axis=2)
	idxs = (base_points[:, np.newaxis].dot(np.ones((1,48))))+(np.ones((num_key_points,1)).dot(lcs_displacement_index[np.newaxis, :])) - 1
	mean_image.ravel(order='F')
	lcs_mean = mean_image.ravel(order='F')[idxs.astype('int')]
	lcs_std =  std_image.ravel(order='F')[idxs.astype('int')]
	lcs_features = np.hstack((lcs_mean, lcs_std))
	lcs_features = lcs_features.reshape(npool_row, npool_col, -1)
	idxs = np.argwhere(np.ones((npool_row, npool_col))).reshape(npool_row, npool_col, 2)
	lcs_features = np.concatenate((lcs_features, idxs), axis=2)
	return lcs_features

if __name__ == "__main__":
	stride = 4
	patch_size = 24
	pixel_border = 16
	test_img = scipy.misc.imread('test.png')
	t = time.time()
	lcs_features = lcs(test_img, stride, patch_size, pixel_border)
	e = time.time()

	matlab_lcs = np.array([float(x) for x in open("lcs_matlab.csv", "r").read().strip().replace("\n", ",").split(",")])
	print(matlab_lcs)
	print(lcs_features.ravel("F"))

	print(np.argwhere((np.abs(matlab_lcs - lcs_features.ravel(order="F")) > 1e-1)))
	assert(np.mean(np.abs(matlab_lcs - lcs_features.ravel(order="F"))) < 1e-1)
	assert(np.max(np.abs(matlab_lcs - lcs_features.ravel(order="F"))) < 1e-1)




