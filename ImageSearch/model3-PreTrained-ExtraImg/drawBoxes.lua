------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'image'
require 'sys'

function drawBoxes(imgOrg, boxes, picked)
  local outString = ''
  --print (boxes)
  local boxThickness = 10
  for loopBoxes = 1, picked:size(1) do
    local startX = math.max(1,(boxes[picked[loopBoxes]])[1])
    local startY = math.max(1,(boxes[picked[loopBoxes]])[2])
    local endX = math.max(1,(boxes[picked[loopBoxes]])[3])
    local endY = math.max(1,(boxes[picked[loopBoxes]])[4])
    outString = outString..tostring(math.floor(startX))..'_'..tostring(math.floor(startY))..'_'..tostring(math.floor(endX))..'_'..tostring(math.floor(endY))..'N'
    --print('coord:..'..startX..' '..startY..' '..endX..' '..endY)
    imgOrg[{ {}, {startY, endY}, {startX, startX+boxThickness} }] = 0
    imgOrg[{ {}, {startY, endY}, {endX-boxThickness, endX} }] = 0
    imgOrg[{ {}, {startY, startY+boxThickness}, {startX, endX} }] = 0
    imgOrg[{ {}, {endY-boxThickness, endY}, {startX, endX} }] = 0
    --print('Confidence score: '..scores[picked[loopBoxes]])
  end
  return imgOrg, outString
end
