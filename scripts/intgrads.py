import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.to('cuda')
    print('model loaded, loading tokeniser')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('tokeniser loaded')
    return model, tokenizer

chosen_model = 'small' #'small'or'large'

if chosen_model == 'small':
    model = "allenai/OLMo-1B-hf" #little model, runs on laptop
else:
    model = "allenai/OLMo-7B-0724-Instruct-hf" #large model, runs on HPC

print('loading model')
olmo, tokenizer = load_model(model)
print('model ready for set up')
print(olmo.device)


p = [json.loads(x) for x in open('../data/prepped_questions.json').read().split('\n')][0]

lig = LayerIntegratedGradients(olmo, olmo.model.embed_tokens)

llm_attr = LLMGradientAttribution(lig, tokenizer)

c=1
n=10000
with open(f'../data/outputs/{chosen_model}_{n}_intgrads.json', 'w') as fp:
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
