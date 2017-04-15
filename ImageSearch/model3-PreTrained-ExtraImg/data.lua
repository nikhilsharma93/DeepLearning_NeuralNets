------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'image'
require 'sys'
require 'os'
require 'lfs'


function ls(path) return sys.split(sys.ls(path),'\n') end

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  'Loading the dataset..' .. sys.COLORS.black ..'\n')

local currentDir = lfs.currentdir()
local lastSlash = string.find(string.reverse(currentDir), "/")
local rootDir = string.sub(currentDir, 1, string.len(currentDir) - lastSlash + 1)

local posDir = rootDir..'HumanBodyData_2/UpperBodyComplete/'
local negDir = rootDir..'HumanBodyData_2/Background/'

--Extra Images
local posDir2 = rootDir..'HumanBodyData_2/UpperBodyCHK_2/'
local negDir2 = rootDir..'HumanBodyData_2/Background1/'

local numberExtraBg = 30000
local totalNoImages = #ls(posDir) + #ls(negDir)
local totalNoImages2 = #ls(posDir2) + numberExtraBg

local allImages = torch.Tensor(totalNoImages+totalNoImages2, 3, 46, 46)
local allLabels = torch.Tensor(totalNoImages+totalNoImages2)

classes = {'body','backg'}

--Loading the Upper Body Images
print ('Loading the Upper Body Images..')
local loopVar1 = 0
for imgName in lfs.dir(posDir) do
	ok,img=pcall(image.load, posDir..imgName)
  if ok then
    allImages[loopVar1+1] = image.load(posDir..imgName)
    allLabels[loopVar1+1] = 1
    loopVar1 = loopVar1 + 1
  end
end

print ('Loading extra Upper Body Images..')
for imgName in lfs.dir(posDir2) do
	ok,img=pcall(image.load, posDir2..imgName)
  if ok then
    allImages[loopVar1+1] = image.load(posDir2..imgName)
    allLabels[loopVar1+1] = 1
    loopVar1 = loopVar1 + 1
  end
end

print ('Loaded '..loopVar1..' Upper images')
--Loading the Background Images
print ('Loading the Background Images..')
for imgName in lfs.dir(negDir) do
	ok,img=pcall(image.load, negDir..imgName)
	if ok then
    allImages[loopVar1+1] = image.load(negDir..imgName)
    allLabels[loopVar1+1] = 2
    loopVar1 = loopVar1 + 1
  end
end

print ('Loading the extra Background Images..')
local extraBgCount = 1
for imgName in lfs.dir(negDir2) do
  if (extraBgCount <= numberExtraBg) then
  	ok,img=pcall(image.load, negDir2..imgName)
  	if ok then
      allImages[loopVar1+1] = image.load(negDir2..imgName)
      allLabels[loopVar1+1] = 2
      loopVar1 = loopVar1 + 1
      extraBgCount = extraBgCount + 1
      print ('extrabg: '..extraBgCount)
    end
  end
end


print('\n'..sys.COLORS.red .. 'Loaded '..loopVar1..' images' .. sys.COLORS.black .. '\n')



----------------------------------------------------------------------
--Data shuffling
local labelsShuffle = torch.randperm((#allLabels)[1])
local portionTrain = 0.8 -- 80% is train data, rest is test data
local trainSize = torch.floor(labelsShuffle:size(1)*portionTrain)
local testSize = labelsShuffle:size(1) - trainSize

-- create train set:
trainData = {
   data = torch.Tensor(trainSize, 3, 46, 46),
   labels = torch.Tensor(trainSize),
   size = function() return trainSize end
}
--create test set:
testData = {
      data = torch.Tensor(testSize, 3, 46, 46),
      labels = torch.Tensor(testSize),
      size = function() return testSize end
}

for i=1,trainSize do
   trainData.data[i] = allImages[labelsShuffle[i]]:clone()
   trainData.labels[i] = allLabels[labelsShuffle[i]]
end
for i=trainSize+1,testSize+trainSize do
   testData.data[i-trainSize] = allImages[labelsShuffle[i]]:clone()
   testData.labels[i-trainSize] = allLabels[labelsShuffle[i]]
end


--Clear memory and delete allImages and allLabels
allImages = nil
allLabels = nil


----------------------------------------------------------------------
print(sys.COLORS.red ..  'Preprocessing the data..' .. sys.COLORS.black ..'\n')
local channels = {'r','g','b'}

print ('Global Normalization\n')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '\nVerify Statistics:' ..sys.COLORS.black .. '\n')


for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '\nVisualization..' ..sys.COLORS.black .. '\n')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   --image.save('/home/nikhil/myCode/learning/Torch/AI2/model3-PreTrained-ExtraImg/trainImages.jpg', first128Samples)
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
   --image.save('/home/nikhil/myCode/learning/Torch/AI2/model3-PreTrained-ExtraImg/testImages.jpg', first128Samples)
end


return {
  trainData = trainData,
  testData = testData,
  mean = mean,
  std = std,
  classes = classes
}
