require 'nn'
require 'cunn'
require 'cudnn'
require 'model/common_dense'
--require 'model/DenseConnectLayer'

local function createModel(opt)

    local actParams = {}
    actParams.actType = opt.act
    actParams.l = opt.l
    actParams.u = opt.u
    actParams.alpha = opt.alpha
    actParams.negval = opt.negval


    function addLayer(model, nChannels, opt)
       if opt.optMemory >= 3 then
          model:add(nn.DenseConnectLayerCustom(nChannels, opt))
       else
          model:add(DenseConnectLayerStandard(nChannels, opt))
       end
    end
    function DenseConnectLayerStandard(nChannels, opt)
       local net = nn.Sequential()
       net:add(nn.SpatialConvolution(nChannels, 1024, 3, 3, 1, 1, 1, 1))
       net:add(nn.ReLU(true))
net:add(nn.SpatialConvolution(1024, 32, 3, 3, 1, 1, 1, 1))
       net:add(nn.ReLU(true))

       return nn.Sequential()
          :add(nn.Concat(2)
             :add(nn.Identity())
             :add(net))
    end
    local function addDenseBlock(model, nChannels, opt, N)
       -- opt.growthRate: output feature number of each conv layer
       for i = 1, N do
          addLayer(model, nChannels, opt)
          nChannels = nChannels + 32
       end
       return nChannels
    end
    -- build one DenseBlock
    function buildDenseBlockUnit(opt)
       local scaleRes = (opt.scaleRes and opt.scaleRes ~= 1) and opt.scaleRes or false
       local nDenseConv  = opt.nDenseConv -- conv layer number in one dense block
       local nChannels = opt.nFeaSDB        -- input feature number opt.growthRate
       local BlockUnit = nn.Sequential()
       nChannels = addDenseBlock(BlockUnit, nChannels, opt, nDenseConv)
       -- local feature fusion via 1x1 conv layer
       BlockUnit:add(nn.SpatialConvolution(nChannels, opt.conv_1x1_count, 1,1, 1,1, 0,0))
       return BlockUnit
    end
 local body = nn.Sequential()
    body:add(nn.SpatialConvolution(opt.nFeat, opt.nFeaSDB, 3,3, 1,1, 1,1))
    body:add(buildDenseBlockUnit(opt))
        body:add(nn.SpatialConvolution(opt.conv_1x1_count, opt.nFeaSDB, 3, 3, 1, 1, 1, 1))

    ret = seq():add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
    if opt.globalSkip then
        -- global skip connection
        ret:add(addSkip(body, true))
    else
        ret:add(body)
    end
    ret:add(upsample_wo_act(opt.scale[1], opt.upsample, opt.nFeat))
        :add(conv(opt.nFeat,opt.nChannel, 3,3, 1,1, 1,1))
    print(ret)

    params, gradParams = ret:getParameters()
    print(params:size(1))
    print(gradParams:size(1))

    return ret

end
return createModel

                      
