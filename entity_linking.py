import os
import json
import spacy
import pickle
from tqdm import tqdm 
import requests
import pandas as pd
nlp = spacy.load("en_core_web_sm")

with open('./OFA/caption_fb_dataset.pickle', 'rb') as h:
    caption = pickle.load(h)

print(caption)

# exit(0)
loef = os.listdir('./entity_json')
ll = []
for i in loef:
    candidate = i.split('.')
    if len(candidate)==3:
        ll.append(candidate[0]+'.0.json')
    else:
        ll.append(i)

loef = list(set(ll))

print(loef)
print(len(ll))
print(len(loef))


import sys
import os
sys.path.append('transformers/examples/research_projects/visual_bert')
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
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast
import pandas as pd
from tqdm import tqdm
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'


start=0
end = 6990

#string -> Visual entity 1 : Desc of visual entity 1 [SEP] text entity 1: text entity desc1 [SEP] caption
#text entities retrieved from both meme text and caption

prohibited_keywords = ['Photograph', 'Photography', 'Getty Images', 'Image', 'iStock', 'stock.xchng', 'Illustration', 'Royalty-free', 'Portrait', 'Alamy']

import jsonlines
train_set = []
with jsonlines.open('../../data/train.jsonl') as reader:
  for obj in tqdm(reader):
    train_set.append(obj)

print(train_set)

# exit(0)
inside_counter = 0
id2text = {i['id']:(i['text'],i['label'],i['img'].split('/')[-1]) for i in train_set}
# counter = 0
# visual_entities = {}
# exceptions = []
# for i in id2text:
#     id = i
#     split_img_path= id2text[i][2].split('.')
#     path_to_search = split_img_path[0]+'.'+'json'
#     print(path_to_search)
#     entities = []
#     try:
#         f=open(os.path.join(os.getcwd(),'entity_json/'+path_to_search))
#         k = json.load(f)
#         for i in range(len(k['webEntities'])):
#             # only proceed if description field is present
#             if 'description' in k['webEntities'][i]:
#                 #if only one word
#                 if len(k['webEntities'][i]['description'].split(' '))==1:
#                     entities.append(k['webEntities'][i]['description'])
#                 else:
#                     doc = nlp(k['webEntities'][i]['description'])
#                     for ent in doc.ents:
#                         # print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                         entities.append((ent.text, ent.label_))



#     except KeyError:
#         #no visual entity is there, in that case retrieve best guess label

#         entities.append(k['bestGuessLabels'])


        
#     except:
#         try:
#             path_to_search1 = split_img_path[0]+'.0.'+'json'
#             path_to_search2 = split_img_path[0]+'.1.'+'json'
#             f1=open(os.path.join(os.getcwd(),'entity_json/'+path_to_search1))
#             k1 = json.load(f1)
#             f2=open(os.path.join(os.getcwd(),'entity_json/'+path_to_search2))
#             k2 = json.load(f2)
#             print('k1',k1, 'k2', k2)
#             for i in range(len(k1['webEntities'])):
#                 # only proceed if description field is present
#                 if 'description' in k1['webEntities'][i]:
#                     #if only one word
#                     if len(k1['webEntities'][i]['description'].split(' '))==1:
#                         entities.append(k1['webEntities'][i]['description'])
#                     else:
#                         doc = nlp(k1['webEntities'][i]['description'])
#                         for ent in doc.ents:
#                             # print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                             entities.append((ent.text, ent.label_))

#             for i in range(len(k2['webEntities'])):
#                 # only proceed if description field is present
#                 if 'description' in k2['webEntities'][i]:
#                     #if only one word
#                     if len(k2['webEntities'][i]['description'].split(' '))==1:
#                         entities.append(k2['webEntities'][i]['description'])
#                     else:
#                         doc = nlp(k2['webEntities'][i]['description'])
#                         for ent in doc.ents:
#                             # print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                             entities.append((ent.text, ent.label_))
#         except KeyError:
#             entities.append(k['bestGuessLabels'])

#         except Exception as e:
#             exceptions.append(e)
#             inside_counter+=1
#         counter+=1
#     print(entities)
#     print(i)
#     visual_entities[id2text[id][2]] = entities
# print(counter)
# print(inside_counter)
# print(exceptions)
# assert inside_counter==0
# print(visual_entities)
# torch.save(visual_entities, './visual_entities.pt')


visual_entities = torch.load('./visual_entities.pt')

cnt = 0
for i in visual_entities:
    temp_ent = []
    loe = visual_entities[i]
    for entity in loe:
        # print(entity)
        if type(entity)!=str:
            
            if entity[0] not in prohibited_keywords:
                print('tuple', entity)
                temp_ent.append(entity)
        else:
            
            if entity not in prohibited_keywords:
                print('string', entity)
                temp_ent.append(entity)
    print(temp_ent)
    print('*********************************************************')
    visual_entities[i] = temp_ent
    # cnt+=1
    # if cnt==5:
    #     break

print(len(visual_entities))
print(len(caption))
# exit(0)
API_ENDPOINT = 'https://www.wikidata.org/w/api.php'

cnt = 0
knowledge_base = []
for i in visual_entities:
    visual_ents = visual_entities[i]
    tmp_desc = []
    counter = 0
    full_str = ''
    for entity in visual_ents:
        if len(entity)==2:
            entity = entity[0]
            params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'search': entity
            }
        else:
            params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'search': entity
            }
        resp = requests.get(API_ENDPOINT,params=params)
        resp = resp.json()
        # print(resp)
        display = resp['search'][0]['display']['label']['value']
        description = resp['search'][0]['display']['description']['value']
        # desc = '[KB{}] '.format(counter) + display + ' are ' + description + '.'
        desc = '[KB] ' + display + ' are ' + description + '.'
        full_str+=desc+' '
        counter+=1
    full_str+= ' [CAPTION] '+caption[i]
    print(full_str)
    cnt+=1
    if cnt==4:
        break
    print('******************')





# # exit(0)
# TAGS = {}
# cnt=0
# bad_names = []
# for i_ in tqdm(loef):
#     f = open(os.path.join(os.getcwd(),'entity_json/'+i_))
#     k = json.load(f)
#     # print('*****************************')
#     # print(caption[i_.split('.')[0]+'.png'])
    
#     # print(k['bestGuessLabels'])
    
#     try:

#         ners = []
#         for i in range(len(k['webEntities'])):
            
#             try:
#                 doc = nlp(k['webEntities'][i]['description'])
#                 for ent in doc.ents:
#                     # print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                     ners.append((ent.text, ent.label_))

#             except:
#                 break
#         if len(i_.split('.'))==3:
#             TAGS[i_.split('.')[0]+'.'+i_.split('.')[2]] = ners
#         else:
#             TAGS[i_.split('.')[0]+'.'+i_.split('.')[1]] = ners

#         # print(TAGS)
#     except:
#         print('in except')
#         if len(i_.split('.'))==3:
#             bad_names.append(i_.split('.')[0]+'.'+i_.split('.')[2])
#         else:
#             bad_names.append(i_.split('.')[0]+'.'+i_.split('.')[1])


#     f.close()
#     # cnt+=1
#     # if cnt==15:
#     #     break

# # print(TAGS)
# print('total in except {}'.format(len(bad_names)))
# print(bad_names)

# PROMPTS = {}
# for i_ in tqdm(caption):

#     fname = i_.split('.')[0]+'.json'
#     # print(fname)
#     if fname in bad_names:
#         continue
    
    
#     tags = [i[0] for i in TAGS[fname]]
#     tags_ = 'I have entities like'
#     for i in tags:
#         tags_+= ' '+i+','

#     tags = tags_
#     # print(tags)
#     PROMPTS[i_] = tags

# with open('prompt_fb.pickle', 'wb') as handle:
#     pickle.dump(PROMPTS, handle, protocol=pickle.HIGHEST_PROTOCOL)



# preset = """
#     Sentence: a man sitting in a chair talking on a cell phone
#     I have entities like Stephen Hawking, A Brief History of Time: From the Big Bang to Black Holes.
#     How to rewrite the sentence using the above entities?
#     Answer: Stephen Hawking who wrote A Brief History of Time: From the Big Bang to Black Holes, sitting in a chair talking on a cell phone.

#     Sentence: a man and woman sitting in bed
#     I have entities like Hitler, Mein-kämpf, Nazi party
#     How to rewrite the sentence using the above entities?
#     Answer: Hitler who was the leader of nazi party and wrote Mein-kämpf is sitting in bed

#     Sentence: a woman holding a sign in front of her face
#     I have entities like Alexandria Ocasio-Cortez, Politics, Unites States
#     How to rewrite the sentence using the above entities?
#     Answer: Alexandria Ocasio-Cortez who is a politician from United States is holding a sign in front of her face
#     """

# with open('prompt_fb.pickle', 'rb') as handle:
#     tags = pickle.load(handle)


# TAGS = []


# with open('./OFA/caption_fb_dataset.pickle', 'rb') as h:
#     caption = pickle.load(h)

# cnt=0
# exp={}
# ids_w_pos_class = []
# import jsonlines
# with jsonlines.open('../../data/train.jsonl') as reader:
#     for obj in reader:
#         if int(obj['label'])==1:
#             ids_w_pos_class.append(obj['id'])

# print(ids_w_pos_class)
# print(len(ids_w_pos_class))

# ids_w_pos_class = list(map(lambda x: str(x)+'.png',ids_w_pos_class))
# print(ids_w_pos_class)
# ids=[]
# prompt=[]
# # exit(0)
# for i in tags:
#     print(i)
#     if tags[i]!='I have entities like' and i in ids_w_pos_class: # should be positive class and have entities
#         txt = "I have an image which is captioned as '{}'. {}'. please rewrite the caption using that entities?".format(caption[i],tags[i])
#         ids.append(i)
#         prompt.append(txt)
        
#         # {}
#         # How to rewrite the sentence using the above entities?
#         # Answer:
#         # .format(caption[i],tags[i])
        
#         # print(preset+txt)
#         # text = preset+txt
#         # try:
#         #     resp = requests.post(
#         #     "https://api.ai21.com/studio/v1/j1-jumbo/complete",
#         #     headers={"Authorization": "Bearer 00ajs6vF5h0JPNYDj7nL4xgHepvJDqum"},
#         #     json={
#         #         "prompt": text, 
#         #         "maxTokens": 25,
#         #         "minTokens": 10,
#         #         "temperature": 0,
#         #         "stopSequences": ['.','\n']
#         #     }
#         #     )
#         # except:
#         #     with open('./fb-rich_{}.pickle'.format(cnt), 'wb') as handle:
#         #         pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)


#         # print(resp.json()['completions'][0]['data']['text'])
#         # exp[i] = resp.json()['completions'][0]['data']['text']
#         cnt+=1

# pd.DataFrame({'id':ids, 'prompt':prompt}).to_csv('prompt_for_rich_caption_fb.csv')




# print(cnt)
# with open('./fb-rich_{}.pickle'.format(cnt), 'wb') as handle:
#     pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed
# import os
# import torch
# import pandas as pd
# import pickle

# os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers

# model_name = "google/flan-t5-xxl"

# ds_config = {
#     "fp16": {
#         "enabled": False,
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "offload_param": {
#             "device": "cpu",
#             "pin_memory": True
#         },
#         "stage3_param_persistence_threshold": 4e5, # Tune this value depending on the capacity of your GPU. With the current value, the GPU memory will peak at ~24GB.
#     },
#     "train_batch_size": 1,
# }

# _ = HfDeepSpeedConfig(ds_config)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# print(model.config)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("Model and tokenizer loaded")

# # inputs = tokenizer.encode("Review: this is the best cast iron skillet you will ever buy. Is this review positive or negative?", return_tensors="pt")

# text = preset+txt







# inputs = tokenizer.encode(text, return_tensors="pt")


# inputs = inputs.to("cuda:0")

# deepspeed_engine, _, _, _ = deepspeed.initialize(
#     model=model,
#     config_params=ds_config,
#     model_parameters=None,
#     optimizer=None,
#     lr_scheduler=None
# )

# deepspeed_engine.module.eval()





# with torch.no_grad():
#     outputs = deepspeed_engine.module.generate(inputs,do_sample=True,temperature=0.01,top_p=0.98,max_length=22)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# print("FINISHED")

# print('***************************************************************************88')



