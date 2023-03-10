# IMPORTANT!!!
#export LD_LIBRARY_PATH=/home1/ekbal_asif/baban/multimodal/cuda/cuda/lib64
#export CUDA_VISIBLE_DEVICES=2

import sys
sys.path.append('transformers/examples/research_projects/visual_bert')
import os
print('right file execution')
#os.environ["LD_LIBRARY_PATH"] = "/home1/ekbal_asif/baban/multimodal/cuda/cuda/lib64"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
from IPython.display import Image, display
import PIL.Image
import io
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast, BertModel
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'

kb_fb = torch.load('./kb_fb.pt')

import jsonlines
train_set = []
with jsonlines.open('../../data/train.jsonl') as reader:
  for obj in tqdm(reader):
    train_set.append(obj)

print(train_set)

# exit(0)

id2text = {i['id']:(i['text'],i['label'],i['img'].split('/')[-1]) for i in train_set}


with open('./explanation-fb_remaining.pickle', 'rb') as handle:
    fb1 = pickle.load(handle)


with open('./explanation-fb-0_2749.pickle', 'rb') as handle:
    fb2 = pickle.load(handle)


explanations = {}
labels = []
src_txt = []
img_paths = []
exps = []


k1,k2 = 0,0
for meme in id2text:
    if id2text[meme][1]==0:
        explanations[meme] = 'this meme is not offensive.'
    else:
        try:
            
            explanations[meme] = fb1[meme].lower().strip().split('.')[0]+'.'
            k1+=1
        except:
            
            explanations[meme] = fb2[meme].lower().strip().split('.')[0]+'.'
            k2+=1


print(explanations)
print(len(explanations))

src_id = []
for meme in id2text:
    src_txt.append(id2text[meme][0])
    src_id.append(meme)
    img_paths.append('../../data/img/'+id2text[meme][2])
    labels.append(id2text[meme][1])
    exps.append(explanations[meme])

print(src_txt[7899:7905])
print(img_paths[7899:7905])
print(labels[7899:7905])
print(exps[7899:7905])
print(k1+k2)



with open('./prompt_fb.pickle', 'rb') as handle:
    prompts = pickle.load(handle)
with open('./caption_fb_dataset.pickle', 'rb') as h:
    caption = pickle.load(h)



print(len(caption))
print(len(prompts))



rich_captions = {}
count = 0
for memes in tqdm(id2text):
    
    try:
        
      img_path = './data/'+id2text[meme][2]
      txt = id2text[memes][0]
      tags = prompts[str(memes)+'.png']
      capt = caption[str(memes)+'.png']
      PROMPT = 'this meme is described as {}. {}.'.format(capt,'This meme contains'+tags[6:-1])
      rich_captions[memes] = PROMPT
    except Exception as e:
      print(e)
      rich_captions[memes] = capt
      count+=1

rich_caps = [rich_captions[i] for i in id2text]
print(rich_caps[7899:7905])
print(count)




train_src  =  pd.DataFrame({'src': src_txt})
train_id = pd.DataFrame({'id': src_id})
train_img = pd.DataFrame({'img': img_paths})
# import pickle
# rich_caps = open("./OFA/rich_caption.pickle", "rb")
# rich_captions = pickle.load(rich_caps)
# rich_caps.close()

# print(len(train_img.keys()))
# dummy = []
# cnt = 0
# discarded_imgs = []
# for i in list(train_img['img']):
#     try:
#         dummy.append(rich_captions[i])
#     except:
#         cnt+=1
#         discarded_imgs.append(i)
#         dummy.append('a picture inside a meme')



train_tgt = pd.DataFrame({'tgt': exps}).applymap(str).apply(lambda x: x.str.lower())

train_offn  =  pd.DataFrame({'offn': labels})



# exps1 = open("./explanation-2000_2621.pickle", "rb")
# exp1 = pickle.load(exps1)
# exps1.close()

# exps2 = open("./explanation-3000_4270.pickle", "rb")
# exp2 = pickle.load(exps2)
# exps2.close()


# exps3 = open("./explanation-0_1164.pickle", "rb")
# exp3 = pickle.load(exps3)
# exps3.close()

# exps4 = open("./explanation-5000_6224.pickle", "rb")
# exp4 = pickle.load(exps4)
# exps4.close()



# rich_caps = open("./OFA/rich_caption.pickle", "rb")
# rich_captions = pickle.load(rich_caps)
# rich_caps.close()

# print(len(train_img.keys()))
# dummy = []
# cnt = 0
# discarded_imgs = []
# for i in list(train_img['img']):
#     try:
#         dummy.append(rich_captions[i])
#     except:
#         cnt+=1
#         discarded_imgs.append(i)
#         dummy.append('a picture inside a meme')



train_conc = pd.DataFrame({'tgt': rich_caps}).applymap(str).apply(lambda x: x.str.lower())





# train_tgt = pd.DataFrame({'tgt': dummy}).applymap(str).apply(lambda x: x.str.lower())


train_src_ , train_img_, train_tgt_, train_offn_, train_conc_, train_id_ = train_src , train_img, train_tgt, train_offn, train_conc, train_id


def find_files(filename, search_path):
   result = False
   filename = filename.split('/')[-1]
   # print(filename)
   # Wlaking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result=True
         break
   return result

# cnt=0
# for i in tqdm(list(train_img['img'])):
#     # print('here')
#     # print(i)
#     if find_files(i, '../../data/img/'):
#         cnt+=1
#     else:
#         print(i)

# print(len(train_img),cnt)

# exit(0)


train_src = train_src_.sample(frac=0.8,random_state=200)
train_img = train_img_.sample(frac=0.8,random_state=200)
train_tgt = train_tgt_.sample(frac=0.8,random_state=200)
train_offn = train_offn_.sample(frac=0.8,random_state=200)
train_conc = train_conc_.sample(frac=0.8,random_state=200)
train_id = train_id_.sample(frac=0.8,random_state=200)


test_src=train_src_.drop(train_src.index)
test_img=train_img_.drop(train_img.index)
test_tgt=train_tgt_.drop(train_tgt.index)
test_offn=train_offn_.drop(train_offn.index)
test_conc = train_conc_.drop(train_conc.index)
test_id = train_id_.drop(train_id.index)


print(train_src[0:10])
print(train_img[0:10])
print(train_tgt[0:10])
print(train_offn[0:10])
print(train_conc[0:10])
print(train_id[0:10])


print(test_src[0:10])
print(test_img[0:10])
print(test_tgt[0:10])
print(test_offn[0:10])
print(test_conc[0:10])
print(test_id[0:10])



# exit(0)

k = str(test_src['src'][2]) in list(train_src['src'][:])
print(k)



print(len(train_src))
print(len(train_img))
print(len(train_tgt))
print(len(train_offn))

# print(cnt)
# print(discarded_imgs)

import os

def find_files(filename, search_path):
   result = False
   filename = filename.split('/')[-1]
   # print(filename)
   # Wlaking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result=True
         break
   return result
# cnt=0
# for i in list(train_img['img']):
#     # print('here')
#     # print(i)
#     if find_files(i, '../../data/img/'):
#         cnt+=1
#     else:
#         print(i)

# print(len(train_img),cnt)


# cnt=0
# for i in list(test_img['img']):
#     # print('here')
#     # print(i)
#     if find_files(i, '../../data/img/'):
#         cnt+=1
#     else:
#         print(i)

# print(len(test_img),cnt)


caps = []
IMAGE_FOLDER='./memotion_dataset_7k/images/'

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.MODEL.device = device
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

frcnn.eval()

# with open('./train_remaining NMT.txt', 'r', encoding="utf-8") as nmt:
#   nmt_op = nmt.readlines()

# # nmt_op = nmt_op.split('\n')
# print(nmt_op[44])
# # nmt_op = list(map(lambda x: x.encode('utf-8'), nmt_op))


# train_src =pd.DataFrame({'src': nmt_op})

print(len(train_tgt))
l = len(train_src)
print(l)
print(train_src)


print(train_img)
def get_img_embedding(img_paths):
	images, sizes, scales_yx = image_preprocess(img_paths) # img_paths -> list of image paths
	output_dict = frcnn(
	  images,
	  sizes,
	  scales_yx=scales_yx,
	  padding="max_detections",
	  max_detections=frcnn_cfg.max_detections,
	  return_tensors="pt",
	)
	features = output_dict.get("roi_features")
	visual_embeds = features
	visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
	visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
	return visual_embeds,visual_token_type_ids,visual_attention_mask

from transformers import BertTokenizer, VisualBertForPreTraining
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler



# from transformers.models.rag.modeling_rag import RagSequenceForGeneration
# from transformers import RagTokenizer, RagRetriever, RagModel
# import torch

# rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# rag_retriever = RagRetriever.from_pretrained(
#     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
# )
# rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq").to('cuda')



# contexts = {}
# lll = len(test_img)


# ID2LAB = torch.load('./id2lab.pt')
# print(ID2LAB)
# id2lab = {}
# for i in ID2LAB:
#     id2lab[i.item()] = ID2LAB[i]

# print(id2lab)
# cnt=0
# for index in tqdm(range(lll)):
#     # try:
#     # ml = test_offn.iloc[index]['offn']
#     frac=0.5
    
#     # ml = list(np.random.binomial(1, frac, size=lll))
#     # print(ml)
#     mc = test_conc.iloc[index]['tgt']
#     mt = test_src.iloc[index]['src']
#     # print(test_id.iloc[index]['id'])
#     # np.random.seed(42)
#     chance = np.random.binomial(1,0.001)
#     print(chance)
    
#     try:
#         if chance:
#             ml = test_offn.iloc[index]['offn']^1
#             cnt+=1
#         else:
#             ml = test_offn.iloc[index]['offn']

#     except:
#         print('in except')
#         ml = 0
    
#     if ml==0:
#         ml = 'not offensive'
#     else:
#         ml = 'offensive'

#     prompt = "Meme Context: "+mc+';\n'+"Meme Text: "+mt+';\n'+"Q: Why is this meme "+ml+'?\n'
#     # prompt = "Meme Context: "+mc+';\n'+"Meme Text: "+mt+';\n'+"Q: Why is this meme offensive or not offensive?\n"
#     # prompt = "Meme Context: "+mc+';\n'+"Meme Text: "+mt+';\n'+"Q: What are the documents supporting this meme?\n"
#     print(prompt)
#     rag_inputs = rag_tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length = 64)
#     with torch.no_grad():
#         question_hidden_states = rag_model.question_encoder(rag_inputs['input_ids'].to('cuda'))[0]
#     #retrieve
#     # 2. Retrieve
#     docs_dict = rag_retriever(rag_inputs["input_ids"].cpu().numpy(), question_hidden_states.cpu().numpy(), return_tensors="pt")
#     # print(rag_tokenizer.batch_decode(docs_dict['context_input_ids'],skip_special_tokens=True))
#     # docs_dict['context_input_ids'][docs_dict['context_input_ids']==2025] = 2000
#     docs_dict['context_input_ids'][docs_dict['context_input_ids']==2555] = 2000
#     docs_dict['context_input_ids'][docs_dict['context_input_ids']==45] = 2000
#     doc_scores = torch.bmm(
#     question_hidden_states.cpu().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
#     ).squeeze(1)
#     with torch.no_grad():
#         outputs = rag_model(
#             context_input_ids=docs_dict["context_input_ids"].to('cuda'),
#              context_attention_mask=docs_dict["context_attention_mask"].to('cuda'),
#             doc_scores=doc_scores.to('cuda')
#         )

#     # batch["encoder_doc_context"] = outputs.generator_enc_last_hidden_state
#     # batch["encoder_doc_attn_mask"] = docs_dict["context_attention_mask"]
#     epsilon = 0.001
#     p = list(outputs.generator_enc_last_hidden_state.detach().cpu().size())
#     # op1 = outputs.generator_enc_last_hidden_state.detach().cpu() + epsilon*np.random.randn(p[0],p[1],p[2])
#     op1 = outputs.generator_enc_last_hidden_state.detach().cpu()
#     op2 = docs_dict["context_attention_mask"].detach().cpu()
#     print(op1.size())
#     print(op2.size())

#     contexts[str(test_img.iloc[index]['img'])] = (op1,op2)
    

# torch.save(contexts, './doc_ctx_wiki_dpr_test_fb_wolab.pt')

# print(cnt/lll)
# exit(0)

# contexts = torch.load('./doc_ctx_wiki_dpr_test_fb.pt')
contexts = torch.load('./doc_ctx_wiki_dpr_test_fb_wolab.pt')

class CustomDataset(Dataset):

    def __init__(self, dataframe1,dataframe2,dataframe3, tokenizer, max_len, start, end):
        self.tokenizer = tokenizer
        self.tokenizer_hi = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data1 = dataframe1[start:end]
        self.data2 = dataframe2[start:end]
        self.img = dataframe3[start:end]
        self.max_len = max_len
        self.data_offn = test_offn[start:end]
        self.id = test_id[start:end]
        # self.data_offn = train_offn[start:end]
    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        #print(index)
        #print(self.data1.iloc[index]['src'])
        batch = {}
        inputs = self.tokenizer_hi(self.data1.iloc[index]['src'], padding="max_length", truncation=True, max_length=self.max_len)
        #print(inputs)
        outputs = self.tokenizer_hi(self.data2.iloc[index]['tgt'], padding="max_length", truncation=True, max_length=self.max_len)
        # visual_embeds,visual_token_type_ids,visual_attention_mask = get_img_embedding(IMAGE_FOLDER+str(self.img.iloc[index]['img'])+".jpg")
        try:
            visual_embeds,visual_token_type_ids,visual_attention_mask = get_img_embedding(str(self.img.iloc[index]['img']))
        except:
            print(str(self.img.iloc[index]['img']))
            visual_embeds,visual_token_type_ids,visual_attention_mask = get_img_embedding(str(self.img.iloc[22]['img']))


        
        batch["input_ids"] = torch.tensor(inputs.input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(inputs.attention_mask,dtype= torch.long)
        #batch["decoder_input_ids"] = torch.tensor(outputs.input_ids,dtype= torch.long)
        batch["decoder_attention_mask"] = torch.tensor(outputs.attention_mask,dtype=torch.long)
        batch["labels"] = outputs.input_ids.copy()
        batch.update({
          "visual_embeds": torch.squeeze(visual_embeds),
          "visual_token_type_ids": torch.squeeze(visual_token_type_ids),
          "visual_attention_mask": torch.squeeze(visual_attention_mask)
        })
        #print(batch["labels"])
        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        batch["exp"] = batch["labels"]
        batch["labels"] = torch.tensor([-100 if token == tokenizer.pad_token_id else token for token in batch["labels"]],dtype=torch.long)

        # k = 0
        # if self.data_offn.iloc[index]['offn']=='not_offensive':
        #     k = 0
        # elif self.data_offn.iloc[index]['offn']=='slight':
        #     k = 1
        # elif self.data_offn.iloc[index]['offn']=='very_offensive':
        #     k=2
        # else:
        #     k=3
        k = int(self.data_offn.iloc[index]['offn'])
        batch["encoder_offn"] = int(k)
        batch["id"] = int(self.id.iloc[index]['id'])
        o1,o2 = contexts[str(self.img.iloc[index]['img'])]

        # batch["encoder_doc_context"] = outputs.generator_enc_last_hidden_state
        # batch["encoder_doc_attn_mask"] = docs_dict["context_attention_mask"]

        batch["encoder_doc_context"] = o1
        batch["encoder_doc_attn_mask"] = o2

        kb_str = kb_fb[str(self.img.iloc[index]['img'].split('/')[-1])]

        # batch["encoder_doc_context"] = kb_str

        kg_ctx = self.tokenizer_hi(kb_str, padding="max_length", truncation=True, max_length=self.max_len)

        batch["kg_input_ids"] = torch.tensor(kg_ctx.input_ids, dtype=torch.long)
        batch["kg_attention_mask"] = torch.tensor(kg_ctx.attention_mask,dtype= torch.long)

        return batch





# ds = CustomDataset(train_src,train_tgt,train_img,tokenizer,28,0,l-1) #train set
# ds_val = CustomDataset(train_src,train_tgt,train_img,tokenizer,28,0,100) # val set

# cnt=0
# for i in ds:
#     print(i)
#     cnt+=1
#     if cnt==4:
#         break

# exit(0)
from transformers import EncoderDecoderModel

from transformers.modeling_outputs import Seq2SeqLMOutput
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


from torch import nn
class EncDecModel(EncoderDecoderModel):

  def __init__(self,config,encoder,decoder):
    super().__init__(config,encoder,decoder)
    self.linear_relu_stack = torch.nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Linear(128, 4),
            nn.Linear(128, 2),
        )
    # self.post_init()
    self.dec2enc_attn = torch.nn.TransformerDecoderLayer(d_model=768, nhead=4, batch_first=True)
    self.kg_model = BertModel.from_pretrained("bert-base-uncased")
   
   
  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        visual_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_offn=None,
        decoder_doc_context=None,
        decoder_doc_attn_mask=None,
        **kwargs,
    ):
        r"""
        Returns:
        Examples::
            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
            >>> # training
            >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
            >>> model.config.pad_token_id = tokenizer.pad_token_id
            >>> model.config.vocab_size = model.config.decoder.vocab_size
            >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
            >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
            >>> outputs = model(input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits
            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
            >>> # generation
            >>> generated = model.generate(input_ids)
        
       

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        """
        print(kwargs)
        print(input_ids)
        # print(doc_context)
        # doc_context = visual_token_type_ids[1]
        # doc_attn_mask = visual_attention_mask[1]
        # visual_token_type_ids = visual_token_type_ids[0]
        # visual_attention_mask=visual_attention_mask[0]


        # ctx_enc = torch.mean(decoder_doc_context,dim=1)
        # ctx_mask = torch.mean(decoder_doc_attn_mask.float(),dim=1)
        # ctx_mask = (ctx_mask>0.0).long()
        # ctx_mask = ctx_mask^1
        

        # print(ctx_enc)
        # print(ctx_mask)
        # print(ctx_enc.shape)
        # print(ctx_mask.shape)
        # print('XXXXXXXXXXXXXXXXXXXX')



        # memory = self.k(ctx_enc)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print(visual_attention_mask)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds,
                visual_token_type_ids=visual_token_type_ids,
                visual_attention_mask=visual_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            
            )
        # print(self.linear_relu_stack)
        # encoder_hidden_states = encoder_outputs[0]
        # outputs = self.linear_relu_stack(encoder_outputs.pooler_output)
   
        # print(np.argmax(outputs.detach().cpu().numpy(),axis=-1))
        # print(outputs.size())
        print(encoder_offn)
        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        #print(decoder_input_ids)

        # Decode
        #print('in normal vat{}'.format(visual_attention_mask.size()))
        #print('in normal {}'.format(attention_mask.size()))
        flag = 0
        try:
          torch.cat((attention_mask,visual_attention_mask),dim=1)
          # visual_attention_mask=visual_attention_mask[0]
        except:
          print('in exception')
          flag=1
          #print('in exception {}'.format(attention_mask.size()))
          visual_attention_mask = torch.full((8*1, 36), 1).to(device) # 32->BS, 4-> beam

         

        # attn_mask = torch.cat((attention_mask,visual_attention_mask),dim=1).long()

        # print(attn_mask)
        # attn_mask = attn_mask^1
        # print(attn_mask)
        # attn_mask = attn_mask.bool()

        # d = self.decoder_layer(tgt=encoder_hidden_states, memory=memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=ctx_mask.bool())

          
          
        # print(attention_mask)
        # print(visual_attention_mask)
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_hidden_states=d,
            encoder_attention_mask=torch.cat((attention_mask,visual_attention_mask),dim=1), # very imp lime
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            
        )





        # Compute loss independent from decoder (as some shift the logits inside them)
        
        print(return_dict)
        loss = None
        if labels is not None:
            #warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[1]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
            # +loss_fct(outputs,encoder_offn)

        # return_dict=False

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        print(loss)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
  # def prepare_inputs_for_generation(
  #       self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
  #   ):
  #       decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
  #       decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
  #       input_dict = {
  #           "attention_mask": attention_mask,
  #           "decoder_attention_mask": decoder_attention_mask,
  #           "decoder_input_ids": decoder_inputs["input_ids"],
  #           "encoder_outputs": encoder_outputs,
  #           "past_key_values": decoder_inputs["past_key_values"],
  #           "use_cache": use_cache,
  #           "doc_context": kwargs['doc_context'],
  #           'doc_attn_mask': kwargs['doc_attn_mask']
  #       }
  #       return input_dict


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from sacrebleu.metrics import BLEU, CHRF, TER
bleu = BLEU()
chrf = CHRF()
ter = TER()
mdl= EncDecModel.from_encoder_decoder_pretrained('uclanlp/visualbert-vqa-coco-pre','bert-base-uncased').to(device)
mdl.load_state_dict(torch.load('./checkpoint_with_kg/checkpoint-8400/pytorch_model.bin'))
# mdl = EncDecModel('uclanlp/visualbert-vqa-coco-pre','bert-base-uncased').from_pretrained('checkpoint-500').to(device)
print(mdl.config)
mdl.config.decoder_start_token_id = tokenizer.cls_token_id
mdl.config.eos_token_id = tokenizer.sep_token_id
mdl.config.pad_token_id = tokenizer.pad_token_id
mdl.config.vocab_size = mdl.config.decoder.vocab_size
mdl.config.max_length = 56
mdl.config.min_length = 4
mdl.config.no_repeat_ngram_size = 1
mdl.config.early_stopping = True
mdl.config.length_penalty = 1.2
mdl.config.num_beams = 1

mdl.eval()
# tokenizer_hi = BertTokenizer.from_pretrained('bert-base-uncased')
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer_hi.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer_hi.pad_token_id
    label_str = tokenizer_hi.batch_decode(labels_ids, skip_special_tokens=True)
    bleu_score = sacrebleu.corpus_bleu(pred_str,[label_str]).score
    #rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "bleu score": bleu_score
    }

# training_args = Seq2SeqTrainingArguments(
#     predict_with_generate=True,
#     # learning_rate=3e-04,
#     evaluation_strategy="steps",
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     fp16=False, 
#     output_dir="./",
#     logging_steps=1,
#     save_steps=500,
#     eval_steps=95,
#     num_train_epochs = 10,
#     # logging_steps=1000,
#     # save_steps=500,
#     # eval_steps=7500,
#     # warmup_steps=2000,
#     # save_total_limit=3,
# )
# # instantiate trainer
# trainer = Seq2SeqTrainer(
#     model=mdl,
#     tokenizer=tokenizer,
#     args=training_args,
#     #compute_metrics=compute_metrics,
#     train_dataset=ds,
#     eval_dataset=ds_val,
# )
# trainer.train()

x = []
pred = []
act = []
pred_proba = []
import math
l = math.floor(len(test_src)/8)*8

assert l%8==0
# l = 32
ds_test = CustomDataset(test_src[0:l],test_tgt[0:l],test_img[0:l],tokenizer,56,0,l)

# l=32
# ds_test = CustomDataset(train_src[0:l],train_tgt[0:l],train_img[0:l],tokenizer,28,0,320)


tmp = 0
ID2LAB = {}
IDS = []
ACTUALS = []
data = torch.utils.data.DataLoader(ds_test, batch_size=8)
for inputs in data:
  print(tokenizer.batch_decode(torch.stack(inputs['exp']).t().detach().cpu().numpy(),skip_special_tokens=True))
  ACTUALS.extend(tokenizer.batch_decode(torch.stack(inputs['exp']).t().detach().cpu().numpy(),skip_special_tokens=True))
  # exit(0)
  # print(np.asarray(inputs['exp']))
  # print(np.asarray(inputs['exp']).transpose())
  
  # print(tokenizer.batch_decode(np.asarray(inputs['exp']).transpose()))
  # try:
  #   k = mdl.generate(
  #     input_ids=inputs['input_ids'].to('cuda'),
  #     attention_mask=inputs['attention_mask'].to('cuda'),
  #     #token_type_ids=torch.unsqueeze(inputs['token_type_ids'],dim=0),
  #     visual_embeds=inputs['visual_embeds'].to('cuda'),
  #     visual_token_type_ids=inputs['visual_token_type_ids'].to('cuda'),
  #     visual_attention_mask=inputs['visual_attention_mask'].to('cuda'),

  #     decoder_doc_context=inputs['encoder_doc_context'].to('cuda'),
  #     decoder_doc_attn_mask=inputs['encoder_doc_attn_mask'].to('cuda')

  #   )
  # except:
  #     print('in exception')
  #     # gt = list(train_tgt.iloc[0:len(x)].tgt)
  #     # print(sacrebleu.corpus_bleu(x,[gt]).score)

  encoder_outputs = mdl.encoder(
                input_ids=inputs['input_ids'].to('cuda'),
                attention_mask=inputs['attention_mask'].to('cuda'),
                # token_type_ids=token_type_ids,
                visual_embeds=inputs['visual_embeds'].to('cuda'),
                visual_token_type_ids=inputs['visual_token_type_ids'].to('cuda'),
                visual_attention_mask=inputs['visual_attention_mask'].to('cuda'),
                
                )


  encoder_hidden_states = encoder_outputs[0]
  print(encoder_outputs.keys())
  print((encoder_outputs[0].detach().cpu().numpy()==encoder_outputs.last_hidden_state.detach().cpu().numpy()).all())
  #   exit(0)

  kg_input_ids = inputs['kg_input_ids'].to('cuda')
  kg_attention_mask = inputs['kg_attention_mask'].to('cuda')
  kg_op = mdl.kg_model(input_ids=kg_input_ids,attention_mask=kg_attention_mask)
  memory = kg_op['last_hidden_state']
  ctx_mask = kg_attention_mask^1


  # print(ctx_enc)
  # print(ctx_mask)
  # print(ctx_enc.shape)
  # print(ctx_mask.shape)
  # print('XXXXXXXXXXXXXXXXXXXX')



  # memory = mdl.k(ctx_enc)
  attn_mask = torch.cat((inputs['attention_mask'].to('cuda'),inputs['visual_attention_mask'].to('cuda')),dim=1).long()

  print(attn_mask)
  attn_mask = attn_mask^1
  print(attn_mask)
  attn_mask = attn_mask.bool()

  d = mdl.dec2enc_attn(tgt=encoder_outputs.last_hidden_state, memory=memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=ctx_mask.bool())
  lhs_placeholder_enc = encoder_outputs.last_hidden_state
  encoder_outputs.last_hidden_state = d # update last hidden state with 'd': 'd' is the attentive features obtained by updating encoder last hidden state with document context
  # dec_inp_id = torch.tensor([tokenizer.pad_token_id  if token == -100 else token for token in inputs["labels"]],dtype=torch.long).to('cuda')
  dec_inp_id = inputs["labels"][inputs["labels"] == -100] = tokenizer.pad_token_id 
  print(dec_inp_id)
  k = mdl.generate(input_ids=inputs['input_ids'].to('cuda'),
      attention_mask=inputs['attention_mask'].to('cuda'),
      #token_type_ids=torch.unsqueeze(inputs['token_type_ids'],dim=0),
      visual_embeds=inputs['visual_embeds'].to('cuda'),
      visual_token_type_ids=inputs['visual_token_type_ids'].to('cuda'),
      visual_attention_mask=inputs['visual_attention_mask'].to('cuda'),encoder_outputs=encoder_outputs,  decoder_doc_context=inputs['encoder_doc_context'].to('cuda'),
      decoder_doc_attn_mask=inputs['encoder_doc_attn_mask'].to('cuda'))
  print(list(map(lambda t: t.decode('utf-8'), list(map(lambda it: it.encode('utf-8'),ds_test.tokenizer_hi.batch_decode(k, skip_special_tokens=True))))))
  # outputs = mdl.linear_relu_stack(encoder_outputs.pooler_output)
  # d = self.decoder_layer(tgt=encoder_hidden_states, memory=memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=ctx_mask.bool())
        # d = self.dec2enc_attn(tgt=encoder_hidden_states, memory=memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=ctx_mask.bool())
          
  output_hidden_states=True
  print(inputs["labels"])
  print(k)
  print(k.size())
  print(inputs["decoder_attention_mask"])
  gen_input = k.detach().clone()
  gen_attn_mask = k.detach().clone()
  gen_attn_mask[gen_attn_mask != 0] = 1
  print('gen attn mask', gen_attn_mask)
  print('k',k)
  print('gen input',gen_input)
  # decoder_outputs = mdl.decoder(
  #       input_ids=inputs["labels"].to('cuda'),
  #       attention_mask=inputs["decoder_attention_mask"].to('cuda'),
  #       # encoder_hidden_states=encoder_hidden_states,
  #       encoder_hidden_states=d,
  #       encoder_attention_mask=torch.cat((inputs["attention_mask"].to('cuda'),inputs["visual_attention_mask"].to('cuda')),dim=1), # very imp line
  #       inputs_embeds=None,
  #       output_attentions=None,
  #       output_hidden_states=True,
        
        
  #   )
  decoder_outputs = mdl.decoder(
        input_ids=gen_input.to('cuda'),
        attention_mask=gen_attn_mask.to('cuda'),
        # encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states=d,
        encoder_attention_mask=torch.cat((inputs["attention_mask"].to('cuda'),inputs["visual_attention_mask"].to('cuda')),dim=1), # very imp line
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=True,
        
        
    )


  # print('decoder outputs logits', decoder_outputs.logits.size())

  # # print(decoder_outputs)

  # print(output_hidden_states)
  # print('decoder outputs hidden state size', decoder_outputs.hidden_states[-1].size())

  # # print(decoder_outputs.hidden_states)
  # # dec_logits_projected = self.projector(decoder_outputs.logits)

  # dec_logits_projected = mdl.projector(decoder_outputs.hidden_states[-1])

  # # mem_mask = inputs["decoder_attention_mask"]^1
  # mem_mask = gen_attn_mask^1
  # mem_mask = mem_mask.bool()
  # print(lhs_placeholder_enc,dec_logits_projected,attn_mask,mem_mask)
  # mem_mask = mem_mask.to('cuda')
  
  # d2e = mdl.decoder_layer(tgt=lhs_placeholder_enc, memory=dec_logits_projected, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=attn_mask, memory_key_padding_mask=mem_mask)
  # print('here')
  e_pooled = encoder_outputs.pooler_output
  # d2e_pooled = d2e[:,0,:].squeeze(1)

  # outputs = mdl.linear_relu_stack(e_pooled*d2e_pooled)
  outputs = mdl.linear_relu_stack(e_pooled)

  # print('enc hidden state size', encoder_hidden_states.size(), 'd2e size', d2e.size())
  op = np.argmax(outputs.detach().cpu().numpy(),axis=-1)
  log_op = torch.log_softmax(outputs, dim=1)
  pred_proba.append(log_op.detach().cpu().numpy())
  print(op,inputs['encoder_offn'])
  for i, j in zip(list(op), inputs["id"]):
    ID2LAB[j] = i
  pred += list(op)
  act += list(inputs['encoder_offn'].detach().cpu().numpy())

  
  

  

  
  #x = np.concatenate((x,k.detach().cpu().numpy()),0)
  #np.save('tokens_test_{}'.format(tmp),x)
  #print(x.detach().cpu().numpy())
  #print(ds_test.tokenizer_hi.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True)..encode('utf-8'))
  x+= list(map(lambda t: t.decode('utf-8'), list(map(lambda it: it.encode('utf-8'),ds_test.tokenizer_hi.batch_decode(k, skip_special_tokens=True)))))
  IDS+=list(inputs["id"])
  #x += ds_test.tokenizer_hi.batch_decode(k, skip_special_tokens=True).encode('utf-8')
  #print(x)
  print(tmp)
  
  tmp+=1
print(x)
PREDICTED=x
torch.save(ID2LAB, './id2lab.pt')
from sklearn.metrics import f1_score,accuracy_score
imgs = list(test_img['img'][0:l])
imgs.append('F1-macro;act')
IDS.append('F1-macro;act')
x.append('-')
p1 = f1_score(act,pred,average = 'macro')
p2 = accuracy_score(act,pred)

pred_proba = np.asarray(pred_proba)
pred_proba = pred_proba.reshape(pred_proba.shape[0]*pred_proba.shape[1], pred_proba.shape[2])
print(pred_proba.shape)
print(roc_auc_score(act,np.squeeze(np.asarray(pred_proba))[:,1]))
pred.append(p1)
act.append(p2)

print(len(imgs),len(pred),len(act),len(x))

details = pd.DataFrame({
    'imgs': IDS,
    'pred': pred,
    'act': act,
    'explanation': x
})

details.to_csv('./results_expl_fb_wkg.csv')
print(p1,p2)
# print(ACTUALS)
# print(len(ACTUALS))



#gt = list(train_tgt.iloc[0:l].tgt)
print('BLEU ', bleu.corpus_score(PREDICTED,[ACTUALS]))
print('CHRF ', chrf.corpus_score(PREDICTED,[ACTUALS]))
print('TER ', ter.corpus_score(PREDICTED,[ACTUALS]))
