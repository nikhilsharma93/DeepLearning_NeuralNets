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
require 'nnx'

--require 'slidingWindow'
require 'nms'
require 'drawBoxes'

local printInfo = true

--Input parameters
--modelPath = '/home/nikhil/myCode/learning/Torch/AI2/model2/'
modelPath = lfs.currentdir()
modelName = 'model.net'
overlapThresh = 0
baseScales = {   0.19}
threshold = 99
--imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/INRIAPerson/train_64x128_H96/pos/crop001028a.png'
--imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/HumanBodyData_1/croppedINRIA/TrainUpperBody/Uppercrop001710e.png'
--imageToLoad = '/home/nikhil/Downloads/IMG-20160816-WA0028.jpg'
--imageToLoad = '/home/nikhil/Downloads/12957444_10208872901607097_645733357698587390_o.jpg'
--imageToLoad = '/home/nikhil/Downloads/DSC02471.JPG'
--imageToLoad = '/home/nikhil/Downloads/wpid-wallhaven-9384.jpg'

--imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/TestingImages/UpperBody/t10.jpg'
imageToLoad = '/home/nikhil/myCode/learning/Torch/AI2/TestingImages/Objects/image.display_009.png'




function parseFFI(cat1, cat2, cat3, iH, iW, threshold, blobs, scale, scores)
  --loop over pixels
  for y=0, iH-1 do
     for x=0, iW-1 do
       confidenceScores = nn.SoftMax():forward(torch.Tensor{cat1[iW*y+x], cat2[iW*y+x], cat3[iW*y+x]})
       cat1Score = confidenceScores[1]
       cat2Score = confidenceScores[2]
       cat3Score = confidenceScores[3]
       maxScore, posMaxScore = torch.max(confidenceScores, 1)
       --if printInfo then print  (fgScore..'   '..bgScore)
       --if printInfo then print ('Checking '..fgConfidence..' against '..threshold )
        if (maxScore[1]*100 > threshold and posMaxScore[1] == 2) then
          --print (classes[posMaxScore[1]])
          entry = {}
          entry[1] = x
          entry[2] = y
          entry[3] = scale
          table.insert(blobs,entry)
          --if printInfo then print ('scores before: '); if printInfo then print (scores)
          --if printInfo then print ('fg conf: '..fgConfidence)
          scores = torch.cat(scores, torch.Tensor{maxScore[1]*100})
          --if printInfo then print ('scores after: '); if printInfo then print (scores)
      end
    end
  end
  return scores
end






--Load the image and model
if printInfo then print  (sys.COLORS.blue..'\nLoading the Image and Model..'..sys.COLORS.black) end
imageOrg = image.load(imageToLoad)
imageOrg = imageOrg:sub(1,3) --Discard the 4th layer (mask layer)
--imageOrg = image.scale(imageOrg, 640/5, 840/5)
--if printInfo then print  (imageOrg:size())
--image.display(imageOrg)
modelTemp = torch.load(modelPath..'/results/'..modelName)
--model = torch.load('/home/nikhil/myCode/learning/Torch/demos-master/person-detector/model.net')

modelTemp.modules[2].modules[5] = nil -- remove logsoftmax
model = modelTemp.modules[1]:clone() -- split model
classifier1 = modelTemp.modules[2]:clone() -- split and reconstruct classifier
classifier = nn.SpatialClassifier(classifier1)
model:add(classifier)
local detectorHt = 32
local detectorWd = 32
model_fov = detectorHt
model_sub = 4

model = model:type('torch.DoubleTensor')
if printInfo then print  (sys.COLORS.blue..'Loaded'..sys.COLORS.black..'\n') end
if printInfo then print (model) end

imgOrgHt, imgOrgWd = imageOrg:size(2), imageOrg:size(3)
maxDimension = math.max(imgOrgHt, imgOrgWd)
if maxDimension > 600 then
  imageOrg = image.scale(imageOrg, imgOrgWd/(maxDimension/600), imgOrgHt/(maxDimension/600))
  --img = image.scale(img, imgOrg:size(3)/loopPyramid, imgOrg:size(2)/loopPyramid)
end
imgOrgHt, imgOrgWd = imageOrg:size(2), imageOrg:size(3)
minDimension = math.min(imgOrgHt, imgOrgWd)
--if printInfo then print  ('New dim------------------: '..imgOrgHt..'  '..imgOrgWd)
maxScaleFactor = math.floor(imgOrgHt/detectorHt, imgOrgWd/detectorWd)

local endScale = detectorHt/minDimension
scales = {}
for i = 1, #baseScales do
  local scalesTemp = baseScales[i]*detectorHt*maxScaleFactor/minDimension
  if scalesTemp >= endScale then
    --if printInfo then print  ('st: '..scalesTemp)
    table.insert(scales, scalesTemp)
  end
end

--for i = 1, #scales do
--  if printInfo then print  (i..'  '..scales[i])
--end

--Preprocess the image
channels = {'r','g','b'}
mean = {}
std = {}

--image.save(imageToLoad.sub(imageToLoad,1,imageToLoad:len()-4)..'Org.jpg', imageOrg)

for i,channel in ipairs(channels) do
  mean[i] = imageOrg[i][{}][{}]:mean(); std[i] = imageOrg[i][{}][{}]:std()
end

for i,channel in ipairs(channels) do
  imageOrg[i][{}][{}]:add(-mean[i]); imageOrg[i][{}][{}]:div(std[i])
end

for i,channel in ipairs(channels) do
  tmean = imageOrg[{i}]:mean(); tstd = imageOrg[{i}]:std(); --if printInfo then print  (tmean); if printInfo then print (tstd)
end



require 'PyramidPacker'
require 'PyramidUnPacker'
unpacker = nn.PyramidUnPacker(model)
packer = nn.PyramidPacker(model, scales)


--Run Sliding Window with NMS and Image Pyramid, and draw detected boxes
pyramid, coordinates = packer:forward(imageOrg)

multiscale = model:forward(pyramid)
distributions = unpacker:forward(multiscale, coordinates)


rawresults = {}
scores = torch.Tensor{}
classes = {'automobile','dog', 'background'}
-- function FFI:
for i,distribution in ipairs(distributions) do
   local cat1 = torch.data(distribution[1]:contiguous())
   local cat2 = torch.data(distribution[2]:contiguous())
   local cat3 = torch.data(distribution[3]:contiguous())
   scores = parseFFI(cat1, cat2, cat3, distribution[1]:size(1), distribution[1]:size(2), threshold, rawresults, scales[i], scores)
end


detections = {}
for i,res in ipairs(rawresults) do
   local scale = res[3]
   local x = res[1]*model_sub/scale
   local y = res[2]*model_sub/scale
   local w = model_fov/scale
   local h = model_fov/scale
   detections[i] = {x, y, x+w, y+h}
end

--if printInfo then print ('no detect: '..#detections)
--if printInfo then print (detections[136][1])
boxes = torch.Tensor(detections)




if #boxes:size() == 0 then
  if printInfo then print  (sys.COLORS.red..'\nNo Upper Body Found'..sys.COLORS.black..'\n') end
  os.exit()
end
if printInfo then print ('Found '..boxes:size(1)..' boxes before NMS') end


pickedAll = torch.Tensor(boxes:size(1))
for i = 1, boxes:size(1) do
  pickedAll[i] = i
end
imageOutTemp = drawBoxes(imageOrg:clone(), boxes, pickedAll, scores)
image.display(imageOutTemp)

picked = nms(boxes, overlapThresh, scores, printInfo)
imageOut, outString = drawBoxes(imageOrg, boxes, picked)
print (outString)
--Display the output
image.display(imageOut)
--image.save(imageToLoad.sub(imageToLoad,1,imageToLoad:len()-4)..'Combined.jpg', imageOut)
--print (imageOut)