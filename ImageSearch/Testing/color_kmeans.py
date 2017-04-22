# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

import utils
import cv2


def getDominantColor(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


	# reshape the image to be a list of pixels
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	# cluster the pixel intensities
	clt = KMeans(n_clusters = 3)
	clt.fit(image)


	hist = utils.centroid_histogram(clt)

	dominantColorRGB = clt.cluster_centers_[np.argmax(hist)]
	print 'dominant color rgb: ', dominantColorRGB
	dominantColorRGB = dominantColorRGB.astype('uint8')
	dominantColorRGB = np.reshape(dominantColorRGB, (1,1,3))
	dominantColorLAB = cv2.cvtColor(dominantColorRGB, cv2.COLOR_RGB2LAB)
	dominantColorLAB = dominantColorLAB[0][0].astype('float64')
	dominantColorLAB[0] *= 100/255.0
	dominantColorLAB[1] -= 128
	dominantColorLAB[2] -= 128

	return dominantColorLAB
