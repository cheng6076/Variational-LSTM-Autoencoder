require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'rnn'
require 'util.misc'
require 'util.Maskh'
require 'util.MaskedLoss'
require 'util.Sampler'
require 'util.KLDCriterion'
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-model','model checkpoint to use')
cmd:option('-data', 'dataset/test', 'dataset to use')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)

local protos = checkpoint.protos
local opt2 = checkpoint.opt
local idx2word, word2idx = checkpoint.vocab[1], checkpoint.vocab[2]

-- Inference: read a sentence, and see what it generates
function eval(x, rf)
  local seq_length = x:size(2)
  local tmp_sentence = torch.zeros(1, seq_length)
  tmp_sentence[1][1] = word2idx['START']
  if opt.gpuid >= 0 then 
    x = x:float():cuda() 
    tmp_sentence = tmp_sentence:float():cuda()
  end
  m = x:clone():fill(1)
  local init_state = {}
  for L=1,opt2.num_layers do
    local h_init = torch.zeros(1, opt2.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
  end
  local enc_state = {[0] = clone_list(init_state)}

  for t=1,seq_length do
    protos.enc:evaluate()
    local lst = protos.enc:forward{x[{{},t}], m[{{},t}], unpack(enc_state[t-1])}
    enc_state[t] = {}
    for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
  end
  protos.connection_enc:evaluate()
  protos.connection_dec:evaluate()
  local mean, log_var = unpack(protos.connection_enc:forward(enc_state[seq_length][#init_state]))
  local z = protos.sampler:forward({mean, log_var})
  local dec_first = protos.connection_dec:forward(z)
  local dec_state = {[0] = enc_state[seq_length]}
  dec_state[0][#init_state]:copy(dec_first)
  for t=1,seq_length-1 do
    protos.dec:evaluate()
    local lst = protos.dec:forward{tmp_sentence[{{},t}], unpack(dec_state[t-1])}
    dec_state[t] = {}
    for i=1,#init_state do table.insert(dec_state[t], lst[i]) end
    local predictions = lst[#lst]
    local maxs, indices = predictions:max(2)
    tmp_sentence[{{},t+1}]:copy(indices)
  end
  --writing file
  for k=2, seq_length do
    rf:write(idx2word[x[1][k]])
    rf:write(' ')
  end
  rf:write('\n')
  for k=2, seq_length do
    rf:write(idx2word[tmp_sentence[1][k]])
    rf:write(' ')
  end
  rf:write('\n\n')
end

f = io.open(opt.data, 'r')
rf = io.open(opt.data .. 'result', 'w')
for line in f:lines() do
  local sentence = {}
  table.insert(sentence, word2idx['START'])
  for word in line:gmatch'([^%s]+)' do
    if word2idx[word]~=nil then table.insert(sentence, word2idx[word]) end
  end
  if #sentence>2 then
    sentence = torch.Tensor(sentence)
    sentence = sentence:view(1, -1)
    eval(sentence, rf)
  end
end
rf:close()
f:close()
print ('Done!')
