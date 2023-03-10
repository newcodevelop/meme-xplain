import transformers

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

from tqdm.auto import tqdm
import pandas as pd
import jsonlines
import requests


class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr( 
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )


class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J




config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# gpt.to(device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# gpt.to(device)

# gpt = nn.DataParallel(gpt)
# model = nn.DataParallel(model)
def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)

add_adapters(gpt)
gpt.to(device)
# gpt = gpt.to(device)



# prompt = tokenizer("A cat sat on a mat", return_tensors='pt')
# prompt = {key: value.to(device) for key, value in prompt.items()}
# out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)
# # out = gpt.module.generate(**prompt, min_length=128, max_length=128, do_sample=True)
# print(tokenizer.decode(out[0]))

# prompt = """a meme is described as three young men sitting at a table looking at a cell phone. This meme have entities like "cell phones". The meme text reads: "indian fans in real life indian fans on facebook". This meme is offensive because """


# prompt = tokenizer(prompt, return_tensors='pt')
# prompt = {key: value.to(device) for key, value in prompt.items()}
# out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)
# # out = gpt.module.generate(**prompt, min_length=128, max_length=128, do_sample=True)
# print(tokenizer.decode(out[0]))

# exit(0)


batch = pd.read_csv('./chatgpt_request_response.tsv',sep='\t')
batches = ['<|startoftext|> Prompt: '+str(list(batch['prompts'])[i])+'. Response: '+str(list(batch['response'])[i])+'<|endoftext|>' for i in range(len(batch))]


from datasets import load_dataset
from bitsandbytes.optim import Adam8bit

# gpt.gradient_checkpointing_enable()


# optimizer = Adam8bit(gpt.parameters(), lr=1e-5)

# with torch.cuda.amp.autocast():
#     for epochs in range(10):
#       for row in batches:
#           if len(row) <= 1:
#               continue

#           batch = tokenizer(row, truncation=True, max_length=212, return_tensors='pt')
#           batch = {k: v.cuda() for k, v in batch.items()}
#           # print(batch)

#           out = gpt.forward(**batch,)

#           loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
#                                 reduction='mean')
#           print(loss)
#           loss.backward()

#           optimizer.step()
#           optimizer.zero_grad()
#       print('*********')



eos_token_id = tokenizer("<|endoftext|>", return_tensors='pt')

print('eos token id {}'.format(eos_token_id))


# prompt = """<|startoftext|> Prompt: I have a meme which can described as a man wearing a white hat with a suit and tie. Meme contains entities like Army, Getty Images. The meme text reads: "we have to exterminate all gays". Tell in brief why is this meme considered offensive. Response:"""

# prompts = batches[9]+'\n\n'+batches[10]+'\n\n'+batches[11]+'\n\n'+prompt

# prompt = tokenizer(prompts, return_tensors='pt')
# prompt = {key: value.to(device) for key, value in prompt.items()}
# out = gpt.generate(**prompt, min_length=128, max_length=1048, do_sample=True, temperature=0.1)
# print(tokenizer.decode(out[0]))

train_set = []
with jsonlines.open('../../data/train.jsonl') as reader:
  for obj in tqdm(reader):
    train_set.append(obj)

id2text = {i['id']:(i['text'],i['label']) for i in train_set}

import pickle
with open('./prompt_fb.pickle', 'rb') as handle:
    prompts = pickle.load(handle)
with open('./caption_fb_dataset.pickle', 'rb') as h:
    caption = pickle.load(h)


k = 0
# for memes in id2text:
#   try:
#     if id2text[memes][1]==1: # if label is offensive
#       img_path = './data/'+str(memes)+'.png'
#       txt = id2text[memes][0]
#       tags = prompts[str(memes)+'.png']
#       capt = caption[str(memes)+'.png']
#       PROMPT = '<|startoftext|> Prompt: I have a meme which is described as {}. {}. {} Tell in brief why is this meme considered offensive. Response:'.format(capt,'This meme contains'+tags[6:-1],txt)
#       print('************')
#       print(PROMPT)
#       print('-----------')
#       prompt_gpt = batches[9]+'\n\n'+batches[10]+'\n\n'+batches[11]+'\n\n'+PROMPT
#       prompt_gpt = tokenizer(prompt_gpt, return_tensors='pt')
#       prompt_gpt = {key: value.to(device) for key, value in prompt_gpt.items()}
#       out = gpt.generate(**prompt_gpt, min_length=128, max_length=1048, do_sample=True, temperature=0.1)
#       print(tokenizer.decode(out[0]))
#       print('************')

#       if k==2:
#         break
#       k+=1
#   except:
#     print('jjjjjjjj')



with open('./cluster_fb_best_k.pkl', 'rb') as handle:
  cluster = pickle.load(handle)


cluster2meme = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
for i in range(7):
  for j in cluster:
    if cluster[j]==i:
      cluster2meme[i].append(j)


import random
random.seed(10)

map_idx_to_clst = {0:[0,1,2,3], 1:[4,5,6,7,8,9], 2:[10,11,12], 3:[13,14,15,16,17,18,19], 4: [20,21,22,23], 5:[24,25,26,27,28], 6:[29,30,31]}


idx_to_op = {}




# stop_words_ids = [tokenizer.encode(stop_word, add_prefix_space = False) for stop_word in ["\n", ".\n"]]

stop_words_ids = [tokenizer.encode(stop_word) for stop_word in ["<|endoftext|>",".", "\n","\n\n","\n\n\n","\n\n\n\n","\n\n\n\n\n", ".\n"]]

print(stop_words_ids)
# exit(0)
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = [[21943], [5657]])])

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_words_ids)])
print(len(id2text))
cnt=0

offn_memes = 0
for memes in id2text:
    if id2text[memes][1]==1:
        offn_memes+=1

print(offn_memes)
# exit(0)

exp = {}

for memes in tqdm(id2text):
    try:
        if id2text[memes][1]==1: # if label is offensive
          img_path = './data/'+str(memes)+'.png'
          txt = id2text[memes][0]
          tags = prompts[str(memes)+'.png']
          capt = caption[str(memes)+'.png']
          PROMPT = '<|startoftext|> Prompt: I have a meme which is described as {}. {}. "{}" Tell in brief why is this meme considered offensive. Response:'.format(capt,'This meme contains'+tags[6:-1],"The meme text reads: "+txt)
          print('************')
          
          # print(PROMPT)
          # print('-----------')
          print(k)
          if memes in cluster2meme[0]:
            cluster_no = 0
          elif memes in cluster2meme[1]:
            cluster_no = 1
          elif memes in cluster2meme[2]:
            cluster_no = 2
          elif memes in cluster2meme[3]:
            cluster_no = 3

          elif memes in cluster2meme[4]:
            cluster_no = 4
          elif memes in cluster2meme[5]:
            cluster_no = 5
          else:
            cluster_no = 6
          # prompt_gpt = ""
          # for i in map_idx_to_clst[cluster_no][0:2]:
          #   prompt_gpt+=batches[i]+'\n\n'
          # prompt_gpt+=PROMPT

          # print(prompt_gpt)
          # try:
          #   resp = requests.post(
          #   "https://api.ai21.com/studio/v1/experimental/j1-grande-instruct/complete",
          #   headers={"Authorization": "Bearer zcRwr5fTCa8cOy003wVyYRthWAQW7Tyk"},
          #   json={
          #       "prompt": prompt_gpt, 
          #       "maxTokens": 50,
          #       "minTokens": 10,
          #       "temperature": 0,
          #       "stopSequences": ['\n']
          #   }
          #   )
          # except Exception as e:
          #   print('Exception 1 is {}'.format(e))
          #   with open('./explanation-fb-0_{}.pickle'.format(k), 'wb') as handle:
          #       pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)

          # print(resp.json()['completions'][0]['data']['text'])
          # exp[memes] = resp.json()['completions'][0]['data']['text']
          # print('-----------')
          # prompt_gpt = batches[9]+'\n\n'+batches[10]+'\n\n'+batches[11]+'\n\n'+PROMPT
          # prompt_gpt = tokenizer(prompt_gpt, return_tensors='pt')
          # prompt_gpt = {key: value.to(device) for key, value in prompt_gpt.items()}
          # out = gpt.generate(**prompt_gpt, min_length=128, max_length=1048, do_sample=True, temperature=0.1,no_repeat_ngram_size=3,eos_token_id=50256)
          # out = gpt.generate(**prompt_gpt, min_length=128, max_length=1048, do_sample=False, repetition_penalty=1.2)
          # print(tokenizer.decode(out[0]))
          # print('************')

          # idx_to_op[memes] = tokenizer.decode(out[0])

          # if k==2:
            # with open('./idx2op_0_500.pickle', 'wb') as handle:
            #     pickle.dump(idx_to_op, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # break
          k+=1
          # print(res)
    except Exception as e:
        prompt_gpt=""
        prompt_gpt+=batches[0]+'\n\n'+batches[1]+'\n\n'+batches[2]+'\n\n'+PROMPT
        
        resp = requests.post(
            "https://api.ai21.com/studio/v1/experimental/j1-grande-instruct/complete",
            headers={"Authorization": "Bearer zcRwr5fTCa8cOy003wVyYRthWAQW7Tyk"},
            json={
                "prompt": prompt_gpt, 
                "maxTokens": 50,
                "minTokens": 10,
                "temperature": 0,
                "stopSequences": ['\n']
            }
            )

        exp[memes] = resp.json()['completions'][0]['data']['text']
        print(resp.json()['completions'][0]['data']['text'])

        print('Exception 2 is {}'.format(e))
        # print('jjjjjjjj')
        # print(resp.json())
        # k+=1
        # cnt+=1
        # if k==2:
        #     break

print(cnt)



print(exp)
with open('./explanation-fb_remaining.pickle', 'wb') as handle:
    pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./idx2op.pickle', 'wb') as handle:
#     pickle.dump(idx_to_op, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('./explanation-fb_remaining.pickle', 'rb') as handle:
    idx2op = pickle.load(handle)

# assert idx_to_op==idx2op

print(idx2op)

