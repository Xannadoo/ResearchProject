import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from classes import QCA
import random

import bitsandbytes as bnb

import spacy
nlp = spacy.load('en_core_web_sm')

import warnings
# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

torch.manual_seed(23)

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

bnb_config = create_bnb_config()
olmo, tokenizer = load_model("allenai/OLMo-1B-hf", bnb_config)

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

    out = output[len(prompt):]
    out2 = output2[len(prompt2):]

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

c = 0
with open('little_model_counterfactuals.json', 'w') as fp:
    for k, v in prompts.items():
        print(v['prompt_o'])
        print(v['gold_o'])
        print(v['prompt_s'])
        print(v['gold_s'])
        out,gold_present,wrong_present,out2,gold2_present,wrong2_present = get_responses(v)
        fp.write(json.dumps({k: {'gold_present': gold_present, 
                                 'wrong_present': wrong_present, 
                                 'gold2_present': gold2_present, 
                                 'wrong2_present': wrong2_present,
                                 'out': out,
                                 'out2':out2}}))
        c+=1
        if c > 10:
            break
        