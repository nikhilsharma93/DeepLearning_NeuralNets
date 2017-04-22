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

local datasetNum = torch.LongStorage({1, 2, 3, 4, 5})

local datasetBaseDir = rootDir..'Images_set2/'
local testDir = rootDir..'Images_set2/test_batch/'
local negDir = rootDir..'Images_set2/background/'
local negTestDir = rootDir..'Images_set2/backTest/'

local totalNoImages = 0
for i = 1, datasetNum:size() do
  local currentDir = datasetBaseDir..'data_batch_'..tostring(datasetNum[i])..'/'
  totalNoImages = totalNoImages + #ls(currentDir)
end
print(totalNoImages)
totalNoImages = totalNoImages  + #ls(testDir) + #ls(negDir) + #ls(testDir)/2


local allImages = torch.Tensor(totalNoImages, 3, 32, 32)
local allLabels = torch.Tensor(totalNoImages)
print ('size all images: ')
print (allImages:size())

--classes = {'airplane', 'automobile', 'cat', 'dog', 'horse'}
classes = {'automobile', 'dog', 'background'}

local tempLabels = {1, 5}
local originalLabels = {}
for i = 1, #tempLabels do
  originalLabels[tempLabels[i]] = i
end
tempLabels = nil
local classImages = torch.LongStorage(#classes):fill(0)

--Loading the Training Images
print ('Loading the datasets..')
local loopVar1 = 0

for i = 1, datasetNum:size() do
  local currentDir = datasetBaseDir..'data_batch_'..tostring(datasetNum[i])..'/'
  print ('Loading from dataset '..tostring(i))
  for imgName in lfs.dir(currentDir) do
  	ok,img=pcall(image.load, currentDir..imgName)
    if (  ok and ( imgName.find(imgName, 'Label1') or imgName.find(imgName, 'Label5') )   ) then
      allImages[loopVar1+1] = image.load(currentDir..imgName)
      local currentLabel = originalLabels[tonumber(imgName:sub(6,6))]
      classImages[currentLabel] = classImages[currentLabel] + 1
      allLabels[loopVar1+1] = currentLabel
      loopVar1 = loopVar1 + 1
    end
  end
end

for imgName in lfs.dir(negDir) do
	ok,img=pcall(image.load, negDir..imgName)
  if ok then
    allImages[loopVar1+1] = image.load(negDir..imgName)
    allLabels[loopVar1+1] = 3
    loopVar1 = loopVar1 + 1
  end
end



local numTraining = loopVar1
print ('Loaded the training data. \n Loaded '..tostring(numTraining)..' samples.\n')
print ('Distribution across classes: ')
for i = 1, #classes do
  print (classes[i]..'  --->  '..tostring(classImages[i])..' samples\n' )
end


--Loading the Test Images
print ('Loading the Test Images..')
print ('loopvar before negtest: '..tostring(loopVar1))
local i = 0
for imgName in lfs.dir(negTestDir) do
  if i < 1000 then
  	ok,img=pcall(image.load, negTestDir..imgName)
    if ok then
      allImages[loopVar1+1] = image.load(negTestDir..imgName)
      allLabels[loopVar1+1] = 3
      loopVar1 = loopVar1 + 1
      i = i + 1
    end
  end
end
print ('loopvar after negtest: '..tostring(loopVar1))


classImages:fill(0)
for imgName in lfs.dir(testDir) do
	ok,img=pcall(image.load, testDir..imgName)
  if (  ok and ( imgName.find(imgName, 'Label1') or imgName.find(imgName, 'Label5') )   ) then
    allImages[loopVar1+1] = image.load(testDir..imgName)
    local currentLabel = originalLabels[tonumber(imgName:sub(6,6))]
    classImages[currentLabel] = classImages[currentLabel] + 1
    allLabels[loopVar1+1] = currentLabel
    loopVar1 = loopVar1 + 1
  end
end


print ('Loaded the testing data. \n Loaded '..tostring(loopVar1 - numTraining)..' samples.\n')
print ('Distribution across classes: ')
for i = 1, #classes do
  print (classes[i]..'  --->  '..tostring(classImages[i])..' samples\n' )
end

print('\n'..sys.COLORS.red .. 'Loaded '..loopVar1..' images in total' .. sys.COLORS.black .. '\n')



----------------------------------------------------------------------
--Data shuffling
--local labelsShuffle = torch.randperm((#allLabels)[1])
--local portionTrain = 0.8 -- 80% is train data, rest is test data
--local trainSize = torch.floor(labelsShuffle:size(1)*portionTrain)
--local testSize = labelsShuffle:size(1) - trainSize

--Data shuffling
local labelsShuffleTrain = torch.randperm(numTraining)
local trainSize = numTraining
local testSize = (#allLabels)[1] - trainSize
print('train and test size: '..tostring(trainSize)..'  '..tostring(testSize))

-- create train set:
trainData = {
   data = torch.Tensor(trainSize, 3, 32, 32),
   labels = torch.Tensor(trainSize),
   size = function() return trainSize end
}
--create test set:
testData = {
      data = torch.Tensor(testSize, 3, 32, 32),
      labels = torch.Tensor(testSize),
      size = function() return testSize end
}

for i=1,trainSize do
   trainData.data[i] = allImages[labelsShuffleTrain[i]]:clone()
   trainData.labels[i] = allLabels[labelsShuffleTrain[i]]
end

testData.data = allImages[{ {trainSize+1, trainSize+testSize}, {}, {} }]
testData.labels = allLabels[{ {trainSize+1, trainSize+testSize} }]
--for i=trainSize+1,testSize+trainSize do
   --testData.data[i-trainSize] = allImages[labelsShuffle[i]]:clone()
   --testData.labels[i-trainSize] = allLabels[labelsShuffle[i]]
--end


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
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end


return {
  trainData = trainData,
  testData = testData,
  mean = mean,
  std = std,
  classes = classes
}
