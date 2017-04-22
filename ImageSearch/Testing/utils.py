from __future__ import division

import numpy as np
import cv2

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar


def getRegionCoords(coords, imageOrg):
	x1 = int(coords[0])
	y1 = int(coords[1])
	x2 = int(coords[2])
	y2 = int(coords[3])
	imgCropped = imageOrg[y1:y2, x1:x2, :]
	cv2.imshow('img1', imgCropped); cv2.waitKey(90000)

	xCenter = (x1+x2)/2
	yCenter = y1 + (1/2)*(y2-y1)
	xWidth = min(46, (x2-x1)/4)
	yWidth = min(46, (y2-y1)/4)

	return [xCenter - xWidth/2, yCenter - yWidth/2, xCenter + xWidth/2, yCenter + yWidth/2]


def getNearestColor(clr, dict):
	minDiff = 500
	for clrName in dict:
		targetClr = np.array(dict[clrName])
		currentDiff = np.linalg.norm(clr - targetClr)
		if currentDiff < minDiff:
			minDiff = currentDiff
			clrTag = clrName
	return clrTag


'''
def rgbToLab(clrs):
	rVal, gVal, bVal = clrs[0], clrs[1], clrs[2]

	var_R = ( rVal / 255 )
	var_G = ( gVal / 255 )
	var_B = ( bVal / 255 )

	if ( var_R > 0.04045 ):
		 var_R = ( ( var_R + 0.055 ) / 1.055 ) ^ 2.4
	else:
		var_R = var_R / 12.92
	if ( var_G > 0.04045 ):
		 var_G = ( ( var_G + 0.055 ) / 1.055 ) ^ 2.4
	else:
		var_G = var_G / 12.92
	if ( var_B > 0.04045 ):
		 var_B = ( ( var_B + 0.055 ) / 1.055 ) ^ 2.4
	else:
		var_B = var_B / 12.92

	var_R = var_R * 100
	var_G = var_G * 100
	var_B = var_B * 100

	X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
	Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
	Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

	var_X = X / Reference-X
	var_Y = Y / Reference-Y
	var_Z = Z / Reference-Z

	if ( var_X > 0.008856 ):
		 var_X = var_X ^ ( 1/3 )
	else:
		var_X = ( 7.787 * var_X ) + ( 16 / 116 )
	if ( var_Y > 0.008856 ):
		 var_Y = var_Y ^ ( 1/3 )
	else:
		var_Y = ( 7.787 * var_Y ) + ( 16 / 116 )
	if ( var_Z > 0.008856 ):
		 var_Z = var_Z ^ ( 1/3 )
	else:
		var_Z = ( 7.787 * var_Z ) + ( 16 / 116 )

	L = ( 116 * var_Y ) - 16
	a = 500 * ( var_X - var_Y )
	b = 200 * ( var_Y - var_Z )
'''
