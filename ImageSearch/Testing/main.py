import subprocess
from color_kmeans import getDominantColor
import cv2
from utils import getRegionCoords
from utils import getNearestColor
import sys


colorDict = {}
colorDict['red'] = [53.2, 80.1, 67.2]
colorDict['blue'] = [32.3, 79.19, -107.86]
colorDict['green'] = [87.73, -86.18, 83.18]
colorDict['white'] = [100, 0.005, -0.01]
colorDict['black'] = [0.0, 0.0, 0.0]
colorDict['gray'] = [65.86, 0.003, -0.007]
colorDict['orange'] = [54.54, 40.2, 58.27]

fileName = 'detectUpperPyramid.lua'
color = 'red'
image = '/home/nikhil/myCode/learning/Torch/AI2/TestingImages/Objects/21.jpg'

detectString = subprocess.check_output(['qlua', fileName, image])
if len(detectString) == 1:
    print '\nNo Upper Body found\n'
    sys.exit()

imageOrg = cv2.imread(image)

imgOrgHt, imgOrgWd = imageOrg.shape[:2]
maxDimension = max(imgOrgHt, imgOrgWd)
if maxDimension > 600:
    imageOrg = cv2.resize(imageOrg, (0,0), fx = 1/(maxDimension/600.0), fy = 1/(maxDimension/600.0))
    #imageOrg = image.scale(imageOrg, imgOrgWd/(maxDimension/600), imgOrgHt/(maxDimension/600))
cv2.imshow('imresized', imageOrg); cv2.waitKey(90000)

detections = detectString.split('N')
del detections[-1]

detections = [x.split('_') for x in detections]

for UpperBodyCoords in detections:
    coords = getRegionCoords(UpperBodyCoords, imageOrg.copy())
    x1 = int(coords[0])
    y1 = int(coords[1])
    x2 = int(coords[2])
    y2 = int(coords[3])
    imgCropped = imageOrg[y1:y2, x1:x2, :]
    cv2.imshow('img', imgCropped); cv2.waitKey(90000)
    dominantColor = getDominantColor(imgCropped)
    print 'DominantColor LAB: ', dominantColor
    colorTag = getNearestColor(dominantColor, colorDict)
    print '\nTag Detected: {} \n'.format(colorTag)

'''
69_55_229_215N91_108_187_204N

img=cv2.imread('/home/nikhil/myCode/learning/Torch/AI2/colorTagging/python-kmeans-dominant-colors/python-kmeans-dominant-colors/images/jp.png')
getDominantColor(img)
'''
