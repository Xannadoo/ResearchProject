## from captum's plot_seq_attr as that didn't let me change the output graph

from textwrap import shorten
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import spacy
nlp = spacy.load('en_core_web_sm')

def plot_seq_attr(
        attr_res, show: bool = False, figsize=(10,3), filename='captum_attributions'):
        """
        Generate a matplotlib plot for visualising the attribution
        of the output sequence.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
        """

        fig, ax = plt.subplots(figsize=figsize)

        data = attr_res.seq_attr.cpu().numpy()

        #fig.set_size_inches(max(data.shape[0] / 2, 6.4), max(data.shape[0] / 4, 4.8))

        shortened_tokens = [
            shorten(t, width=50, placeholder="...") for t in attr_res.input_tokens
        ]
        ax.set_xticks(range(data.shape[0]), labels=[x.replace('Ġ', ' ').replace('Ċ', '\n').replace('âĢĵ', '-') for x in shortened_tokens])

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(
            ax.get_xticklabels(),
            rotation=-90,
            ha="right",
            rotation_mode="anchor",
        )

        fig.set_facecolor("white")

        # pos bar
        ax.bar(
            range(data.shape[0]),
            [max(v, 0) for v in data],
            align="center",
            color="#4772b3",
        )
        # neg bar
        ax.bar(
            range(data.shape[0]),
            [min(v, 0) for v in data],
            align="center",
            color="#d0365b",
        )

        ax.set_ylabel("Sequence Attribution", rotation=90, va="bottom")

        plt.savefig(f'{filename}.png')
        if show:
            plt.show()
            return None  # mypy wants this
        else:
            return fig, ax
        
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

#change & change_q from Sagniks code
change = {'first': 'last', 
          'earlier': 'later', 
          'younger': 'older', 
          'later': 'earlier', 
          'older': 'younger',
          'more recently': 'earlier'}

def make_prompt(C,Q):
    prompt = f"""Use the context to answer the question. Context: {C} Question: {Q} Answer: """
    return prompt

def change_q(question):
    for k, v in change.items():
        if k in question:
            return question.replace(k, v)
        
def make_reversed_qs(Q, A):
    doc = nlp(Q)
    ents = set()
    for ent in doc.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_) 
        if ent.text in Q and ent.label_ in ['PERSON','NORP','FAC','ORG','GPE', 'LOC', 'PRODUCT', 'EVENT','WORK_OF_ART','LAW','LANGUAGE'] :
            ents.add(ent.text) 
    
    #print(ents, Q, end = ' ')
    ents = list(ents)
    if len(ents)==2:
        if ents[0] in ents[1] or ents[1] in ents[0]:
            return None, None
        elif ents[0] in A:
            #print('accepted')
            A2 = ents[1]
        elif ents[1] in A:
            #print('accepted')
            A2 = ents[0]
        else:
            #print('failed')
            return None, None
        Q2 = change_q(Q)
        return Q2, A2

    else:
        #print('failed')
        return None,None