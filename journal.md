# Timeline, meetings and other important notes

* [14th September 2024](#14th-september-2024)
* [15th October 2024](#15th-october-2024)

### 17th October 2024

## Trying to run project...
- Cannot install directly from requirements.txt. Fails because of distutils not being supported. Installing newer versions manually and hoping for the best. 
- Cannot access linked version of neuralcoref, so will try to use [huggingface/neuralcoref](https://github.com/huggingface/neuralcoref) and hope for the best.
- Library jiant hasn't been supported for several years and is being a royal pain due to outdated requirements. Tried manually installing from source code and forcing the requirements, with partial success. Going to change code to avoid using this and will come up with some other workaround if/when this proves impossible.
- paths set to 'QA_SKILLS_HOME' causing crashing, replacing with 'PWD' to resolve.
- Other small changes due to updates to libraries.

- datasetprocessors/run.py is running...


### 15th October 2024

## Plan settled
*Present: Anna and Chrisanna*
- Start with replicating Anna's paper, using gen model OLMo
- Extend after, if successful
- Don't cry.

### 14th September 2024

## Prelimary Meeting on research project ideas
*Present: Anna and Chrisanna*
Discussed paper [Machine Reading, Fast and Slow: When Do Models “Understand” Language?](https://aclanthology.org/2022.coling-1.8/), and [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://aclanthology.org/P19-1334/)
__Idea 1:__ 
- Repeat work from the first paper with a generative model ([OLMo](https://allenai.org/olmo)? open-source, code and training data avaliable) instead.
- Different reasoning type?
- Different interpretability type?

__Idea 2:__
- Look at training data for a model and try to replicate HANS work.
- Look for patterns in data that might suggest tasks that the model would struggle with.
- Data is very messy!! Pulled from several datasets, synthetic data, chatgpt annotation.

