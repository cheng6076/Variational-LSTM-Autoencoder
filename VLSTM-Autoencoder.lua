require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
model_utils = require 'util.model_utils'
BatchLoader = require 'util.BatchLoaderC'
require 'util.MaskedLoss'
require 'util.Maskh'
require 'util.misc'
require 'util.Sampler'
require 'util.KLDCriterion'
encoder = require 'model.encoder'
decoder = require 'model.decoder'
VAE = require 'model.VAE'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', 'corpus', 'path of the dataset')
cmd:option('-sample_dir', 'samples', 'path of the sampled sentences')
cmd:option('-batch_size', 10, 'batch size')
cmd:option('-max_epochs', 50, 'number of full passes through the training data')
cmd:option('-rnn_size', 300, 'dimensionality of sentence embeddings')
cmd:option('-vae_hidden_size', 300, 'dimensionality of vae hidden layer')
cmd:option('-vae_latent_size', 30, 'dimensionality of latent variable')
cmd:option('-word_vec_size', 150, 'dimensionality of word embeddings')
cmd:option('-max_sentence_l', '30', 'maximum number of words in each sentence')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',50,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 2000, 'save when seeing n examples')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-beta1', 0.9, 'momentum parameter 1')
cmd:option('-beta2', 0.999, 'momentum parameter 2')
cmd:option('-learningRate', 0.001, 'learning rate')
cmd:option('-decayRate',0.97,'decay rate for sgd')
cmd:option('-decay_when',0.1,'decay if validation does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 0, 'print batch times')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid + 1)
end
if opt.cudnn == 1 then
  assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
  print('using cudnn...')
  require 'cudnn'
end

-- create data loader
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.max_sentence_l)
opt.seq_length = loader.max_sentence_l - 1  --the decoder input is from SOS to the last word; the decoder output is from the first word to EOS
opt.vocab_size = #loader.idx2word

-- model
protos = {}
protos.enc = encoder.lstm(opt.vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size)
protos.dec = decoder.lstm(opt.vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size)
protos.connection_enc = VAE.get_encoder(opt.rnn_size, opt.vae_hidden_size, opt.vae_latent_size)
protos.sampler = nn.Sampler()
protos.connection_dec = VAE.get_decoder(opt.rnn_size, opt.vae_hidden_size, opt.vae_latent_size)
protos.criterion = nn.MaskedLoss()
protos.KLD = nn.KLDCriterion()
-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- params and grads
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec, protos.connection_enc, protos.connection_dec)
params:uniform(-0.05, 0.05)
print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end
-- encoder initial states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 then h_init = h_init:cuda() end
  table.insert(init_state, h_init:clone())
  table.insert(init_state, h_init:clone())
end

--this extracts a specific layer from a gmodule
function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'enc_lookup' then
      enc_lookup = layer
    elseif layer.name == 'dec_lookup' then
      dec_lookup = layer
    end
  end
end 
protos.enc:apply(get_layer)
protos.dec:apply(get_layer)

--evaluation
function eval_split(split_idx)
  print('evaluating loss over split index ' .. split_idx)
  local n = loader.split_sizes[split_idx]
  loader:reset_batch_pointer(split_idx)
  local loss = 0
  local KLDloss = 0
  local count = 0
  for i = 1,n do
    local x, m = loader:next_batch(split_idx)
    local tmp_sentence = torch.zeros(opt.batch_size, opt.seq_length-1) --store sentence batch
    if opt.gpuid >= 0 then
      x = x:float():cuda()
      m = m:float():cuda()
    end
    labels = x:sub(1,-1,2,-1):clone()
    y = x:sub(1,-1,1,-2):clone() + 1
    x = x + 1
    local enc_state = {[0] = init_state}
    for t=1,opt.seq_length do
      clones.enc[t]:evaluate()
      local lst = clones.enc[t]:forward{x[{{},t}], m[{{},t}], unpack(enc_state[t-1])}
      enc_state[t] = {}
      for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
    end

    local mean, log_var = unpack(protos.connection_enc:forward(enc_state[opt.seq_length][#init_state]))
    local z = protos.sampler:forward({mean, log_var})
    local dec_first = protos.connection_dec:forward(z)
    local KLDloss = KLDloss + protos.KLD:forward(mean, log_var)
    
    local dec_state = {[0] = enc_state[opt.seq_length]}
    dec_state[0][#init_state]:copy(dec_first)
    for t=1,opt.seq_length-1 do
      clones.dec[t]:evaluate()
      local lst = clones.dec[t]:forward{y[{{},t}], unpack(dec_state[t-1])}
      dec_state[t] = {}
      for i=1,#init_state do table.insert(dec_state[t], lst[i]) end
      local predictions = lst[#lst]
      local maxs, indices = predictions:max(2)
      tmp_sentence[{{},t}]:copy(indices)
      local result = clones.criterion[t]:forward({predictions, labels[{{},t}]})
      loss = loss + result[1]
      count = count + result[3]
    end
  end
  return loss/count + KLDloss
end

local init_state_global = clone_list(init_state)
--training
function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()
  -- load data
  local x, m = loader:next_batch(1)
  if opt.gpuid >= 0 then
    x = x:float():cuda()
    m = m:float():cuda()
  end
  labels = x:sub(1,-1,2,-1):clone()
  y = x:sub(1,-1,1,-2):clone() + 1 
  x = x + 1
  local loss = 0
  local count = 0
  -- Forward pass
  -- 1) encoder
  local enc_state = {[0] = init_state_global}
  for t=1,opt.seq_length do
    clones.enc[t]:training()
    local lst = clones.enc[t]:forward({x[{{},t}], m[{{},t}], unpack(enc_state[t-1])})
    enc_state[t] = {}
    for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
  end

  -- 2) VAE layer
  local mean, log_var = unpack(protos.connection_enc:forward(enc_state[opt.seq_length][#init_state]))
  local z = protos.sampler:forward({mean, log_var})
  local dec_first = protos.connection_dec:forward(z)
  local KLDerr = protos.KLD:forward(mean, log_var)

  -- 3) decoder
  local dec_state = {[0] = enc_state[opt.seq_length]}
  dec_state[0][#init_state]:copy(dec_first)
  local predictions = {}
  for t=1,opt.seq_length-1 do
    clones.dec[t]:training()
    local lst = clones.dec[t]:forward({y[{{},t}], unpack(dec_state[t-1])})
    dec_state[t] = {}
    for i=1,#init_state do table.insert(dec_state[t], lst[i]) end
    predictions[t] = lst[#lst]
    local result = clones.criterion[t]:forward({predictions[t], labels[{{},t}]})
    loss = loss + result[1]
    count = count + result[3]
  end

  -- Backward pass
  -- 1) decoder
  local ddec_state = {[opt.seq_length-1] = clone_list(init_state, true)}  
  for t=opt.seq_length-1,1,-1 do
    local doutput_t = clones.criterion[t]:backward({predictions[t], labels[{{},t}]})
    table.insert(ddec_state[t], doutput_t)
    local dlst = clones.dec[t]:backward({y[{{},t}], unpack(dec_state[t-1])}, ddec_state[t])
    ddec_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k > 1 then ddec_state[t-1][k-1] = v end
    end
  end

  -- 2) VAE layer
  local dKLD_mean, dKLD_log_var = unpack(protos.KLD:backward(mean, log_var))
  local dz = protos.connection_dec:backward(z, ddec_state[0][#init_state])
  local dmean, dlog_var = unpack(protos.sampler:backward({mean, log_var}, dz))
  dmean:add(dKLD_mean)
  dlog_var:add(dKLD_log_var)
  local denc_last = protos.connection_enc:backward(enc_state[opt.seq_length][#init_state], {dmean, dlog_var})

  -- 3) encoder
  local denc_state = {[opt.seq_length] = ddec_state[0]}
  denc_state[opt.seq_length][#init_state]:copy(denc_last)
  for t=opt.seq_length,1,-1 do
    local dlst = clones.enc[t]:backward({x[{{},t}], m[{{},t}], unpack(enc_state[t-1])}, denc_state[t])
    denc_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k > 2 then denc_state[t-1][k-2] = v end
    end
  end

  local grad_norm, shrink_factor
  grad_norm = torch.sqrt(grad_params:norm()^2)
  if grad_norm > opt.max_grad_norm then
    shrink_factor = opt.max_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
  end
  return loss/count + KLDerr, grad_params
end

-- start training
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}
local iterations = opt.max_epochs * loader.split_sizes[1]
for i = 1, iterations do
  -- train 
  local epoch = i / loader.split_sizes[1]
  local timer = torch.Timer()
  local time = timer:time().real
  local _, loss = optim.adam(feval, params, optim_state)
  train_losses[i] = loss[1] --loss is a table, we need to flatten it
  
  -- ###The first index of LookupTable is always 0.###
  enc_lookup.weight[1]:zero()
  enc_lookup.gradWeight[1]:zero()
  dec_lookup.weight[1]:zero()
  dec_lookup.gradWeight[1]:zero()
  -- ###

  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_losses[i]))
  end

  -- validate and save checkpoints
  if epoch == opt.max_epochs or i % opt.save_every == 0 then
    print ('evaluate on validation set')
    local val_loss = eval_split(2) -- 2 = validation
    val_losses[#val_losses+1] = val_loss
    local savefile = string.format('%s/model_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_losses = val_losses
    checkpoint.vocab = {loader.idx2word, loader.word2idx}
    print('saving checkpoint to ' .. savefile)
    torch.save(savefile, checkpoint)
  end

  -- decay learning rate
  if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
    if val_losses[#val_losses-1] - val_losses[#val_losses] < opt.decay_when then
      opt.learningRate = opt.learningRate * opt.decayRate
    end
  end

  -- misc
  if i%5==0 then collectgarbage() end
  if opt.time ~= 0 then
     print("Batch Time:", timer:time().real - time)
  end
end

--test_loss = eval_split(3)
--print (string.format("test_loss = %6.4f", test_loss))
