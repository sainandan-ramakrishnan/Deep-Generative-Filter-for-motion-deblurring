require 'nngraph'
require 'nn'
require 'cunn'
require 'cudnn'
require 'densefield'
local utils = require 'super_resolution.utils'
   
function defineGour_net(input_nc, output_nc, ngf,opt)

   local growthRate = opt.growthRate

   --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
   local dropRate = opt.dropRate

   --# channels before entering the first Dense-Block
   local nChannels = 4 * growthRate
   
   --local nChannels_initial = 96
   --compression rate at transition layers
   --local reduction = opt.reduction

   --whether to use bottleneck structures
   local bottleneck = opt.bottleneck

   --N: # dense connected layers in each denseblock
   local N = (opt.depth - 4)/3
   if bottleneck then N = N/2 end


   function addLayer(model, nChannels, opt,j)
      if opt.optMemory >= 3 then
         model:add(nn.DenseConnectLayerCustom(nChannels, opt,j))
      else
         model:add(nn.Concat(2)
            :add(nn.Identity())
            :add(DenseConnectLayerStandard(nChannels, opt)))      
      end
   end

   local function addDensefield(model, nChannels, opt, N)
      for i = 1, N do 
         addLayer(model, nChannels, opt,i)
         nChannels = nChannels + opt.growthRate
      end
      return nChannels
   end


   local function skip_connect(opt,N)
      local inner_skip = nn.Sequential()
      local highway = nn.Identity()
      nChannels = addDensefield(inner_skip, nChannels, opt, N)
      if opt.optMemory >= 3 then inner_skip:add(nn.JoinTable(2)) end  
      inner_skip:add(LastLayer(nChannels, opt))
      --inner_skip:add(nn.Dropout(0.5))
      return nn.Sequential():add(nn.ConcatTable():add(inner_skip):add(highway)):add(nn.CAddTable(true))
      --return inner_skip  
   end
   
      
   -- Build DenseNet
   local model = nn.Sequential() 
   model:add(nn.SpatialConvolution(input_nc, nChannels,3,3,1,1,1,1))   --output:256x256xnChannels = 256x256*(2*gr) 
   --model:add(nn.ReLU(true)) 
   --model:add(nn.SpatialBatchNormalization(nChannels))
   --model:add(nn.SpatialConvolution(nChannels_initial,nChannels,1,1,1,1,0,0))
   --model:add(nn.LeakyReLU(0.2,true))
   model:add(skip_connect(opt,N)) 
   nChannels = 4*growthRate 
   model:add(nn.SpatialConvolution(nChannels, growthRate, 3, 3, 1, 1, 1, 1))
   model:add(nn.LeakyReLU(0.2,true))
   model:add(nn.SpatialConvolution(growthRate,output_nc,3,3,1,1,1,1))
   model:add(nn.Tanh())
   utils.init_msra(model)
   utils.init_BN(model)
   return model
end


function defineD_n_layers(input_nc, output_nc, ndf, n_layers,opt)
        local netD = nn.Sequential() 
        netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2,true))    
        nf_mult = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2,true))
        end
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2,true))
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))       
        netD:add(nn.Sigmoid())
        return netD       
end
