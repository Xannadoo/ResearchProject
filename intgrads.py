import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helpers import plot_seq_attr
from captum.attr import LayerIntegratedGradients, LLMGradientAttribution, TextTokenInput

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

chat = [
    {"role": "system", "content": "You are a helpful bot that uses the provided context to answer questions. You do not answer with any other tokens but the answer entity."},
    {"role": "user", "content": "Context: Ainhoa Artolazábal Royo( born 6 March 1972) is a road cyclist from Spain. She represented her nation at the 1992 Summer Olympics in the women's road race. Allen Holden( 18 April 1911 – 12 December 1980) was a New Zealand cricketer. He played two first- class matches for Otago between 1937 and 1940. Question: Who was born earlier, Allen Holden or Ainhoa Artolazábal?. Answer:"},
    {"role": "assistant", "content": "Allen Holden"}
]

tokenizer.chat_template =  ''

print(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False))

p = [json.loads(x) for x in open('train_comp.json').read().split('\n')][0]

lig = LayerIntegratedGradients(olmo, olmo.model.embed_tokens)

llm_attr = LLMGradientAttribution(lig, tokenizer)

c=1
n=100
with open(f'{chosen_model}_{n}_intgrads.json', 'w') as fp:
    for k, v in p.items():
        Q = v['prompt_o']
        A = v['gold_o']
        Q2 = v['prompt_s']
        A2 = v['gold_s']

        n_steps = 20

        inp = TextTokenInput(Q, tokenizer)

        attr_res = llm_attr.attribute(inp, target=A, n_steps=n_steps)

        data = attr_res.seq_attr.cpu().numpy()

        inp = TextTokenInput(Q2, tokenizer)

        attr_res = llm_attr.attribute(inp, target=A2, n_steps=n_steps)

        data2 = attr_res.seq_attr.cpu().numpy()

        fp.write(json.dumps({k: {'prompt': Q,
                                'answer': A, 
                                'igs': data.tolist(),
                                'prompt2': Q2, 
                                'answer2': A2, 
                                'igs2': data2.tolist()}})+ '\n')
        c+=1
        if c > n:
            break
