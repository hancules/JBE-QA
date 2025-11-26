import os
import sys
import re
import random
import torch
import datasets
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

from evaluate_closed_model import test_closed_model
from evaluate_open_model import test_open_model
from config import dataset_name

def set_seed(seed: int = 20250917):
    random.seed(seed)                     
    np.random.seed(seed)                  
    torch.manual_seed(seed)               
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_open(model_id: str) -> bool:
    closed_models = ['gpt-4.1', 'gpt-4o', 'gpt-5', 'o3', 'o4-mini', 'claude']
    if any(a in model_id for a in closed_models):
        return False
    else:
        return True  
    
def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=None, help='specify the model')
    parser.add_argument('--fewshot', type=str, default='false', help='specify whether to provide examples')
    parser.add_argument('--thinking', type=str, default='false', help='specify whether to provide examples')
    args = parser.parse_args()
    model_id = args.model_id
    fewshot = True if args.fewshot == 'true' else False
    thinking = True if args.thinking == 'true' else False
    
    if is_open(model_id):
        results = test_open_model(model_id, fewshot, dataset_name)
    else:
        results = test_closed_model(model_id, fewshot, thinking, dataset_name)
        
    valid_results = results.query('in_exemplars == False')
    print(
        f"Model: {model_id}",
        f"ACC: {accuracy_score(valid_results['answer'], valid_results['prediction']):.2f}",
        f"F1: {f1_score(valid_results['answer'], valid_results['prediction']):.2f}",
    )
    

if __name__ == "__main__":
    main()