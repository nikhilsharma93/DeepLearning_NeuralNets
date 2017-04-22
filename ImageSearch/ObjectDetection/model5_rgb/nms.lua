------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'sys'


function nms(boxes, overlapThresh, scores, printInfo)
  if printInfo then print (sys.COLORS.blue..'\nPerforming NMS..'..sys.COLORS.black) end

  --Initialize the list of picked boxes
  pick = torch.Tensor{}

  --Get the coordinates of all boxes
  x1 = boxes:index(2,torch.LongTensor{1})
	y1 = boxes:index(2,torch.LongTensor{2})
	x2 = boxes:index(2,torch.LongTensor{3})
	y2 = boxes:index(2,torch.LongTensor{4})

  --Computer the area of bounding boxes and sort based on score
  ones = torch.Tensor(boxes:size(1),1)
  area = torch.cmul((x2-x1+ones),(y2-y1+ones))
  temp1, ind = torch.sort(x2:reshape(x2:size(1)))

  --temp1, ind = torch.sort(scores)

  --Keep looping until indices still remain
  while true do
    last = ind:size(1)
    i = ind[last]
    --print ('\nworking with: '..i)
    pick = torch.cat(pick, torch.Tensor{i})
    suppress = torch.Tensor{last}

    for loopBox = 1, last-1 do
      j = ind[loopBox]
      --print ('checking against: '..j)

      xx1 = math.max(x1[i][1], x1[j][1])
      yy1 = math.max(y1[i][1], y1[j][1])
      xx2 = math.min(x2[i][1], x2[j][1])
      yy2 = math.min(y2[i][1], y2[j][1])

      w = math.max(0, xx2-xx1+1)
			h = math.max(0, yy2-yy1+1)


      --Find the overlap. Criterion used is intersection/union
      overlap = w*h / math.min(area[j][1], area[i][1])
      --print ('\noverlaps: '..overlap..'  '..w*h..' '..area[j][1]..'  '..area[i][1])

      if overlap > overlapThresh then
        --print ('Suppressed')
        --print ('overlap is: '..overlap)
        suppress = torch.cat(suppress, torch.Tensor{loopBox})
      end
    end
    suppress = torch.sort(suppress)
    --print ('Done. suppress is: ')
    --print (suppress)
    --Delete the Suppressed / Encountered indices
    ind = torch.totable(ind)
    for i = suppress:size(1), 1, -1 do
      --print ('i is: '..i)
      table.remove(ind, suppress[i])
    end

    --return if ind is empty
    if #ind == 0 then
      --print('Found '..pick:size(1)..' boxes after NMS')
      return pick
    end

    ind = torch.LongTensor(ind)
  end
  if printInfo then print('Found '..pick:size(1)..' boxes after NMS') end
  return pick
end




--[[
---Sample testing
boxes = {}
scores = {1,20,2,1,1,1}
boxes[1] = {12, 14, 16, 20}
boxes[2] = {24, 84, 152, 212}
boxes[3] = {36, 84, 164, 212}
boxes[4] = {12, 96, 140, 224}
boxes[5] = {24, 96, 152, 224}
boxes[6] = {24, 108, 152, 236}
scores = torch.Tensor(scores)
boxes = torch.Tensor(boxes)
out = nms(boxes, 0.3, scores)
print ('final out:--------- ')
print (out)
]]--
