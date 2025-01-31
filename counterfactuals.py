import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from classes import QCA
import re
from random import randint
import spacy
nlp = spacy.load('en_core_web_sm')
import warnings
# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

torch.manual_seed(23)

def load_model(model_name):
    print('load model internal')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')
    print('model loaded, loading tokeniser')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('tokeniser loaded')
    return model, tokenizer

chosen_model = 'large' #'small'or'large'

if chosen_model == 'small':
    model = "allenai/OLMo-1B-hf" #little model, runs on laptop
else:
    model = "allenai/OLMo-7B-0724-Instruct-hf" #large model, runs on HPC

print('loading model')
olmo, tokenizer = load_model(model)
print('model ready for set up')
print(olmo.device)
chat = [
    {"role": "system", "content": "You should use the provided context to answer the question given. Your reply should contain only the correct answer and nothing more."},
    {"role": "user", "content": "Context: Ainhoa Artolazábal Royo( born 6 March 1972) is a road cyclist from Spain. She represented her nation at the 1992 Summer Olympics in the women's road race. Allen Holden( 18 April 1911 – 12 December 1980) was a New Zealand cricketer. He played two first- class matches for Otago between 1937 and 1940. Question: Who was born earlier, Allen Holden or Ainhoa Artolazábal?"},
    {"role": "assistant", "content": "Allen Holden"}
]

tokenizer.chat_template =  ''

tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
print('chat template applied')

print('model ready to fly, loading dataset')
prompts = [json.loads(x) for x in open('train_comp.json').read().split('\n')][0]
print('dataset loaded')

def get_responses(data):
    prompt = data['prompt_o']
    A = ' ' + data['gold_o']

    prompt2 = data['prompt_s']
    A2= ' ' + data['gold_s']

    #remove words in common
    com = set(A.split()) & set(A2.split())

    A_ = ' '.join([word for word in A.split() if word not in com])
    A2_ = ' '.join([word for word in A2.split() if word not in com])

    if len(A_) < 1 or len(A2_) < 1:
        print('not possible due to common answer strings')
        return A, [], A2, [], 'none' #if one is empty, ie the answer is entirely contained within the other, then this question is not useful to the analysis
    
    else:
        A = A_
        A2 = A2_

    if randint(0,1) == 0:
        r = 'orig'
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs.to(olmo.device)
        response = olmo.generate(**inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

        inputs2 = tokenizer(prompt2, return_tensors='pt', return_token_type_ids=False)
        inputs2.to(olmo.device)
        response2 = olmo.generate(**inputs2, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

        output = tokenizer.batch_decode(response, skip_special_tokens=True)[0].split('\n')[0]
        output2 = tokenizer.batch_decode(response2, skip_special_tokens=True)[0].split('\n')[0]
    else:
        r = 'switched'

        inputs2 = tokenizer(prompt2, return_tensors='pt', return_token_type_ids=False)
        inputs2.to(olmo.device)
        response2 = olmo.generate(**inputs2, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs.to(olmo.device)
        response = olmo.generate(**inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)

        output = tokenizer.batch_decode(response, skip_special_tokens=True)[0].split('\n')[0]
        output2 = tokenizer.batch_decode(response2, skip_special_tokens=True)[0].split('\n')[0]

    out = re.findall(r'(?<=Answer: ).*', output)
    if len(out)>0:
        out = out[0]
    else:
        print(out)
    out2 = re.findall(r'(?<=Answer: ).*', output2)
    if len(out2)>0:
        out2 = out2[0]
    else:
        print(out2)

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
    print()
    return out,gold_present,wrong_present,out2,gold2_present,wrong2_present, r

c=0
n=10000
with open(f'{chosen_model}_{n}_counterfactuals.json', 'w') as fp:
    for k, v in prompts.items():
        print(c)
        out,gold_present,wrong_present,out2,gold2_present,wrong2_present,r = get_responses(v)
        fp.write(json.dumps({k: {'gold_present': gold_present, 
                                 'wrong_present': wrong_present, 
                                 'gold2_present': gold2_present, 
                                 'wrong2_present': wrong2_present,
                                 'out': out,
                                 'out2':out2,
                                 'order': r}})+ '\n')
        c+=1
        if c > n:
            break
print('done!')
