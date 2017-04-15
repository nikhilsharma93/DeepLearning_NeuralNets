------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'image'
require 'sys'
require 'os'
require 'lfs'

require 'slidingWindow'
require 'nms'
require 'drawBoxes'


--Input parameters
modelPath = lfs.currentdir()
modelName = 'modelTest.net'
overlapThresh = 0.6

--imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/INRIAPerson/train_64x128_H96/pos/crop001028a.png'
--imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/HumanBodyData_1/croppedINRIA/TrainUpperBody/Uppercrop001710e.png'
--imageToLoad = '/home/nikhil/Downloads/IMG-20160816-WA0028.jpg'
--imageToLoad = '/home/nikhil/Downloads/12957444_10208872901607097_645733357698587390_o.jpg'
--imageToLoad = '/home/nikhil/Downloads/DSC02471.JPG'
--imageToLoad = '/home/nikhil/Downloads/wpid-wallhaven-9384.jpg'
imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/TestingImages/UpperBody/t11.jpg'
--imageToLoad = '/media/nikhil/nik/VDime/Datasets/upperbodyfrontal_dataset/images/upperbody0024.png'

--Load the image and model
print (sys.COLORS.blue..'\nLoading the Image and Model..'..sys.COLORS.black)
imageOrg = image.load(imageToLoad)
imageOrg = imageOrg:sub(1,3) --Discard the 4th layer (mask layer)
--imageOrg = image.scale(imageOrg, 640/5, 840/5)
print (imageOrg:size())
image.display(imageOrg)
model = torch.load(modelPath..'/results/'..modelName)
--model = torch.load('/home/nikhil/myCode/learning/Torch/demos-master/person-detector/model.net')

model = model:type('torch.DoubleTensor')
print (sys.COLORS.blue..'Loaded'..sys.COLORS.black..'\n')


--Preprocess the image
channels = {'r','g','b'}
mean = {}
std = {}

for i,channel in ipairs(channels) do
  mean[i] = imageOrg[i][{}][{}]:mean(); std[i] = imageOrg[i][{}][{}]:std()
end

for i,channel in ipairs(channels) do
  imageOrg[i][{}][{}]:add(-mean[i]); imageOrg[i][{}][{}]:div(std[i])
end

for i,channel in ipairs(channels) do
  tmean = imageOrg[{i}]:mean(); tstd = imageOrg[{i}]:std(); print (tmean); print(tstd)
end



--Run Sliding Window with NMS and Image Pyramid, and draw detected boxes
boxes, scores = imgSlide(imageOrg, model)
if #boxes:size() == 0 then
  print (sys.COLORS.red..'\nNo Upper Body Found'..sys.COLORS.black..'\n')
  os.exit()
end
print('Found '..boxes:size(1)..' boxes before NMS')
print (scores)

pickedAll = torch.Tensor(boxes:size(1))
for i = 1, boxes:size(1) do
  pickedAll[i] = i
end
imageOutTemp = drawBoxes(imageOrg:clone(), boxes, pickedAll, scores)
image.display(imageOutTemp)

picked = nms(boxes, overlapThresh, scores)
imageOut = drawBoxes(imageOrg, boxes, picked, scores)

--Display the output
image.display(imageOut)
