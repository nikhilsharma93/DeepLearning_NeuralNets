------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'sys'
require 'os'

function imgSlide(imgOrg, model)

  imgOrgHt, imgOrgWd = imgOrg:size(2), imgOrg:size(3)
  detectorHt = 46
  detectorWd = 46
  detectorStrideX = 9
  detectorStrideY = 9
  local numberOfScales = math.max(1, math.floor(math.min(imgOrgHt/(1.5*detectorHt), imgOrgWd/(1.5*detectorWd))))
  local startScale = math.min(math.floor(math.max(imgOrgHt, imgOrgWd)/200), numberOfScales)

  scores = torch.Tensor{}
  boxes = torch.Tensor{}

  --If the image is smaller than the detector itself
  if numberOfScales == 0 then
    print (sys.COLORS.red..'\nCannot run the detector since the image size is less than the detector size of '..detectorHt..'*'..detectorWd..sys.COLORS.black..'\n')
    os.exit()
    --return torch.Tensor{}, torch.Tensor{}
  end


  print (sys.COLORS.blue..'Looping over Image Pyramids..'..sys.COLORS.black)
  --Loop over all scales in the pyramid
  for loopPyramid = startScale, numberOfScales, 1.5 do
    print ('Doing pyramid '..loopPyramid..' out of '.. numberOfScales)
    img = imgOrg:clone()
    img = image.scale(img, imgOrg:size(3)/loopPyramid, imgOrg:size(2)/loopPyramid)
    image.display(img)
    windowStartY = 1; windowEndY = detectorHt;
    while (windowEndY <= img:size(2)) do
        windowStartX = 1; windowEndX = detectorWd;
        while(windowEndX <= img:size(3)) do
          imgToTest = img[{ {}, {windowStartY, windowEndY}, {windowStartX, windowEndX} }]
          modelOp = model:forward(imgToTest)
          local fgScore; local bgScore;
          fgScore = modelOp[1]
          bgScore = modelOp[2]
          fgConfidence = bgScore*100/(fgScore+bgScore)
          local valueOp; local indexOp
          valueOp, indexOp = torch.max(modelOp, 1)
          --if indexOp[1] == 1 then --It is Upper Body
          if (fgConfidence > 99.9 ) then --It is Upper Body
            print ('CONFID: '..fgConfidence..'  '..fgScore..'  '..bgScore)
            local boxStartX = windowStartX * loopPyramid
            local boxStartY = windowStartY * loopPyramid
            local boxEndX = windowEndX * loopPyramid
            local boxEndY = windowEndY * loopPyramid
            boxes = torch.cat(boxes, (torch.Tensor{boxStartX, boxStartY, boxEndX, boxEndY}):reshape(1,4), 1) --Stack them vertically
            scores = torch.cat(scores, torch.Tensor{fgConfidence})
          end
          windowStartX = windowStartX + detectorStrideX
          windowEndX = windowEndX + detectorStrideX
        end
        windowStartY = windowStartY + detectorStrideY
        windowEndY = windowEndY + detectorStrideY
    end
  end
  return boxes, scores
end
