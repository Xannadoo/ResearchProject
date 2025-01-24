import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from classes import QCA
import re

import spacy
nlp = spacy.load('en_core_web_sm')

chosen_model = 'large' #'small' #or 'large'

if chosen_model == 'small':

    model = "allenai/OLMo-1B-hf" #little model, runs on laptop
else:
    model = "allenai/OLMo-7B-0724-Instruct-hf" #large model, runs on HPC

import warnings
# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

torch.manual_seed(23)

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


olmo, tokenizer = load_model(model)

chat = [
    {"role": "system", "content": "You are a helpful bot that uses the provided context to answer questions. You do not answer with any other tokens but the answer entity."},
    {"role": "user", "content": "Context: Ainhoa Artolazábal Royo( born 6 March 1972) is a road cyclist from Spain. She represented her nation at the 1992 Summer Olympics in the women's road race. Allen Holden( 18 April 1911 – 12 December 1980) was a New Zealand cricketer. He played two first- class matches for Otago between 1937 and 1940. Question: Who was born earlier, Allen Holden or Ainhoa Artolazábal?"},
    {"role": "assistant", "content": "Allen Holden"}
]

tokenizer.chat_template =  ''

tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

prompts = [json.loads(x) for x in open('train_comp.json').read().split('\n')][0]

def get_responses(data):
    prompt = data['prompt_o']
    A = ' ' + data['gold_o']

    prompt2 = data['prompt_s']
    A2= ' ' + data['gold_s']

    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
    inputs.to(olmo.device)
    response = olmo.generate(**inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)
    
    inputs2 = tokenizer(prompt2, return_tensors='pt', return_token_type_ids=False)
    inputs2.to(olmo.device)
    response2 = olmo.generate(**inputs2, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

    output = tokenizer.batch_decode(response, skip_special_tokens=True)[0].split('\n')[0]
    output2 = tokenizer.batch_decode(response2, skip_special_tokens=True)[0].split('\n')[0]
    
    out = re.findall(r'(?<=Answer: ).*', output)[0]
    out2 = re.findall(r'(?<=Answer: ).*', output2)[0]

    #check if gold is in output:
    gold_present = False
    for i in A.split():
        if i in out:
            gold_present = True
    
    #also check if the wrong answer is present:
    wrong_present = False
    for i in A2.split():
        if i in out:
            wrong_present = True

    #SAme for the counterfactual
    #check if gold is in output:
    gold2_present = False
    for i in A2.split():
        if i in out2:
            gold2_present = True
    
    #also check if the wrong answer is present:
    wrong2_present = False
    for i in A.split():
        if i in out2:
            wrong2_present = True

    print('*'*30)
    print(f'Output: {out}\nTrue: {A}\ngold in output: {gold_present}, \nwrong in output: {wrong_present}')
    print('*'*30)
    print(f'Output: {out2}\nTrue: {A2}\ngold in output: {gold2_present}, \nwrong in output: {wrong2_present}')
    print('*'*30)
    return out,gold_present,wrong_present,out2,gold2_present,wrong2_present

with open(f'{chosen_model}_counterfactuals.json', 'w') as fp:
    for k, v in prompts.items():
        #print(v['prompt_o'])
        #print(v['gold_o'])
        #print(v['prompt_s'])
        #print(v['gold_s'])
        out,gold_present,wrong_present,out2,gold2_present,wrong2_present = get_responses(v)
        fp.write(json.dumps({k: {'gold_present': gold_present, 
                                 'wrong_present': wrong_present, 
                                 'gold2_present': gold2_present, 
                                 'wrong2_present': wrong2_present,
                                 'out': out,
                                 'out2':out2}}))
        