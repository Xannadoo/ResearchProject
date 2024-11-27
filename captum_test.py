import json
import torch
from classes import QCA
from helpers import plot_seq_attr, create_bnb_config, load_model, make_prompt, change_q, make_reversed_qs

import random
import serpyco

import spacy
nlp = spacy.load('en_core_web_sm')

import warnings
import bitsandbytes as bnb
from captum.attr import LayerIntegratedGradients, LLMGradientAttribution, TextTokenInput
# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

torch.manual_seed(23)

bnb_config = create_bnb_config()
olmo, tokenizer = load_model("allenai/OLMo-1B-hf", bnb_config)
lig = LayerIntegratedGradients(olmo, olmo.model.embed_tokens)
llm_attr = LLMGradientAttribution(lig, tokenizer)
serializer = serpyco.Serializer(QCA)


FILENAME = 'data/validation-spacy.jsonl'
n_steps = 10


qcas = [serializer.load(json.loads(x)) for x in open(FILENAME).read().split('\n')[:-1]]

c = 0
k = 0
prompts = {}
for data in qcas[:]:    
    Q, C, A = data.question.text, data.context.text, data.answer_texts_orig[0]
    Q2, A2 = make_reversed_qs(Q, A)

    if Q2 != None:
        c+=1
        # print(Q, A)
        # print(Q2, A2)
        prompt = make_prompt(C, Q)
        prompt_2 = make_prompt(C, Q2)
        prompts[data.id] = {'prompt_o': prompt, 'gold_o': A, 
                            'prompt_s': prompt_2, 'gold_s': A2}

    else:
        k+=1

print(f'{c}, {k}, {c/(c+k):.2%}')

data = prompts['2wiki-acfbbcd508f311ebbdaaac1f6bf848b6']
prompt = data['prompt_o']
A = data['gold_o']
prompt_2 = data['prompt_s']
A2= data['gold_s']

inp = TextTokenInput(
    prompt, 
    tokenizer)

inp2 = TextTokenInput(
    prompt_2, 
    tokenizer)

attr_res = llm_attr.attribute(inp, target=A, n_steps=n_steps)

attr_res2 = llm_attr.attribute(inp2, target=A2, n_steps=n_steps)

plot_seq_attr(attr_res, show=True, figsize=(40,5), filename=f'img/{A}_prompt_{n_steps}')
plot_seq_attr(attr_res2, show=True, figsize=(40,5), filename=f'img/{A2}_prompt_{n_steps}')