import torch
# necessary for some models
def dummy_compile(fn, *args, **kwargs):
    return fn
torch.compile = dummy_compile

import datasets
import pandas as pd
import gc
import re
import os
import sys

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from sklearn.metrics import accuracy_score

from config import (
    TOKEN,
    COMMIT, 
    PROJECT_DIR,
    format_question,
    exemplar_ids, 
    exemplars, 
    system_prompt, 
)

eos_token_ids = {
    'Nexusflow/Athene-V2-Chat': 151645,
    'Qwen/Qwen2.5-72B-Instruct': 151645,
    "Qwen/Qwen2.5-32B-Instruct": 151645,
    'abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1': 151645,
    'openai/gpt-oss-20b': 200002,
    "openai/gpt-oss-120b": 200002,
    'google/gemma-3-27b-it': 106,
    'google/gemma-3-12b-it': 106,
    'tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3': 128009,
    'tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4': 128009,
    'meta-llama/Llama-3.1-70B-Instruct': 128009,
    "meta-llama/Llama-3.3-70B-Instruct": 128009,
    'llm-jp/llm-jp-3.1-13b-instruct4': 2,
    'llm-jp/llm-jp-3.1-8x13b-instruct4': 2
}

output_patterns = {
    "Nexusflow/Athene-V2-Chat": r"(.*?)<\|im_end\|>", #r"<\|im_start\|>assistant(.*?)<\|im_end\|>",
    "Qwen/Qwen2.5-72B-Instruct": r"(.*?)<\|im_end\|>", 
    "Qwen/Qwen2.5-32B-Instruct": r"(.*?)<\|im_end\|>", 
    "abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1": r"(.*?)<\|im_end\|>", 
    "openai/gpt-oss-20b": r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
    "openai/gpt-oss-120b": r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", #r"<\|start\|>assistant<\|channel\|>final<\|message\|>\s*([01])\s*<\|return\|>"  
    "google/gemma-3-27b-it": r"(.*?)<end_of_turn>", #r"<start_of_turn>model(.*?)<end_of_turn>", 
    "google/gemma-3-12b-it": r"(.*?)<end_of_turn>", 
    "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3": r"(.*?)<\|eot_id\|>", # r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>",
    "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4": r"(.*?)<\|eot_id\|>", 
    "meta-llama/Llama-3.1-70B-Instruct": r"(.*?)<\|eot_id\|>", 
    "meta-llama/Llama-3.3-70B-Instruct": r"(.*?)<\|eot_id\|>",
    "llm-jp/llm-jp-3.1-13b-instruct4": r"(.*?)</s>",  # r"応答:(.*?)</s>", 
    "llm-jp/llm-jp-3.1-8x13b-instruct4": r"(.*?)</s>", 
}
    

class ModelLoader():
    '''A wrapper of tokenizer and model'''
    def __init__(self, model_id, system_prompt=system_prompt):
        self.model_id = model_id
        try:
            self.parse_pattern = output_patterns[model_id]
            self.eos_token_id = eos_token_ids[model_id]
        except KeyError:
            print('Not a valid model id.')
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=TOKEN,
            padding_side='left'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=TOKEN,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.system_prompt = system_prompt
        self.chat_template = self.tokenizer.chat_template
        
        if self.model_id == "meta-llama/Llama-3.1-70B-Instruct": 
            #"tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3",
            #"tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4", 
            #"meta-llama/Llama-3.3-70B-Instruct"
            #]:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = 128009
    
    def add_exemplars(self, fewshot_exemplars) -> list:
        '''Add exemplars to prompt for few-shot learning.'''
        exemplars = []
        if 'llm-jp' not in self.model_id :
            for doc in fewshot_exemplars:
                exemplars += [
                    {"role": "user", "content": format_question(doc)},
                    {"role": "assistant", "content": str(1 if doc['answer'] else 0)},
                ]
        else:
            first_exemplar = fewshot_exemplars[0]
            exemplars = [
                    {"role": "user", "content": self.system_prompt + format_question(first_exemplar)},
                    {"role": "assistant", "content": str(1 if first_exemplar['answer'] else 0)},
                ]
            for idx in range(1, len(fewshot_exemplars)):
                doc = fewshot_exemplars[idx]
                exemplars += [
                    {"role": "user", "content": format_question(doc)},
                    {"role": "assistant", "content": str(1 if doc['answer'] else 0)},
                ]
            
        return exemplars
    
    def parse_output(self, output) -> str:
        '''Parse model output into 0/1/NotANumber.'''
        try:
            pred = re.search(self.parse_pattern, output, re.S).group(1).strip()
            pred = "NotANumber" if pred not in ['0','1'] else int(pred)
        except:
            pred = "NotANumber"
    
        return pred
    
    def make_conversation(self, rows, fewshot_exemplars) -> str:
        '''Convert a dataset instance into a prompt.'''
        messages = []
        for row in rows:
            question = format_question(one_row=row)
            message = [{"role": "system", "content": self.system_prompt}]
            
            if fewshot_exemplars is not None:
                message += self.add_exemplars(fewshot_exemplars)
                query = question
            else:
                query = (self.system_prompt if ('llm-jp' in self.model_id) else '') + question
                
            message += [{"role": "user", "content": query}]
            messages.append(message)
        
        conversation = self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        return conversation

    def predict(self, rows, fewshot_exemplars=None) -> list:
        '''Generate predictions for the given dataset instances(rows).'''
        raw_conversations = self.make_conversation(rows=rows, fewshot_exemplars=fewshot_exemplars)
        tokenized_conversations = self.tokenizer(raw_conversations, return_tensors="pt", padding='longest',padding_side='left')
        
        @torch._dynamo.disable
        def safe_generate(model, **kwargs):
            return model.generate(**kwargs)
        
        self.model.eval()
        with torch.inference_mode():
            outputs = safe_generate(
                self.model,
                input_ids = tokenized_conversations['input_ids'].to(self.model.device),
                attention_mask = tokenized_conversations['attention_mask'].to(self.model.device),
                do_sample=False if 'gpt-oss' not in self.model_id else True,
                top_k=None if 'gpt-oss' not in self.model_id else 50,
                temperature=0 if 'gpt-oss' not in self.model_id else 1.0,
                max_new_tokens=1000 if 'gpt-oss' not in self.model_id else 131072//4,
                tokenizer=self.tokenizer,
                eos_token_id=self.eos_token_id,
            ).cpu()
        
        input_len = int(tokenized_conversations['input_ids'].shape[1])
        genrated_ids = outputs[:, input_len:]
        verbalised_outputs = self.tokenizer.batch_decode(genrated_ids)  
        predictions = [self.parse_output(output=x) for x in verbalised_outputs]
        
        return predictions
            
            
def test_open_model(
    model_id: str, 
    fewshot: bool, 
    dataset_name: str,
) -> pd.DataFrame:
    def get_batch_size(model_id:str, fewshot:bool) -> int:
        '''Determine the batch size for the given model and setting.'''
        if model_id in [
            "openai/gpt-oss-120b", 
            "llm-jp/llm-jp-3.1-8x13b-instruct4"
        ]:
            batch_size = 4 
        elif model_id in [ 
            "openai/gpt-oss-20b", 
            "llm-jp/llm-jp-3.1-13b-instruct4",
            "google/gemma-3-27b-it",
            "google/gemma-3-12b-it" ,
            "abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1"       
        ]:
            batch_size = 32
        else: # over 70b
            batch_size = 16
            
        return batch_size//2 if fewshot else batch_size

    batch_size = get_batch_size(model_id=model_id, fewshot=fewshot)
    model = ModelLoader(model_id) 
    dfs = []
    for split in ['train', 'validation', 'test']:
        dataset_splited = datasets.load_dataset(
            dataset_name,
            token=TOKEN,
            split=f'{split}',
            revision=COMMIT,
        ) 
        dataloader = DataLoader(
            range(len(dataset_splited)), 
            batch_size=batch_size, 
            shuffle=False
        )

        predictions = []
        for batch_ids in tqdm(dataloader, total=len(dataloader)):
            batch = dataset_splited.select(batch_ids)
            prediction = model.predict(
                batch, 
                fewshot_exemplars=exemplars if fewshot else None
            )
            predictions += prediction
        
        df = dataset_splited.to_dict()
        df = pd.DataFrame(df)
        df['in_exemplars'] = [(item in exemplar_ids) for item in df['id']]
        df['prediction'] = [(0 if pred == 'NotANumber' else pred) for pred in predictions]
        df['NotANumber'] = [pred == 'NotANumber' for pred in predictions]
        df['accuracy'] = (df['prediction'] == df['answer'])
        df['split'] = split
        df['model'] = model_id
        df['fewshot'] = fewshot
        dfs.append(df)
        
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    result_dir = PROJECT_DIR / 'results' / 'results_per_model'
    result_dir.mkdir(parents=True, exist_ok=True)
    all_results = pd.concat(dfs).reset_index(drop=True)
    
    return all_results