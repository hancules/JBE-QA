import json
import time
import os
import anthropic
import datasets
import pathlib
import pandas as pd

from openai import OpenAI
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from config import (
    CLAUDE_API_KEY, 
    OPENAI_API_KEY,
    TOKEN,
    COMMIT,
    PROJECT_DIR,
    format_question,
    system_prompt,
    exemplars,
    exemplar_ids
)

def test_closed_model(
        model_id: str, 
        fewshot: bool, 
        thinking: bool, 
        dataset_name: str,
) -> pd.DataFrame:
    if 'claude' in model_id:
        results = test_claude_model(model_id=model_id, fewshot=fewshot, thinking=thinking, dataset_name=dataset_name)
    else:
        results = test_openai_model(model_id=model_id, fewshot=fewshot, dataset_name=dataset_name)
    return results


def test_claude_model(
        model_id: str, 
        thinking: bool, 
        fewshot: bool, 
        dataset_name: str,
) -> pd.DataFrame:
    '''Evaluate a Claude model on jbe-qa dataset, returning a dataframe as results.'''
    def compose_requests(
        model_id, 
        fewshot, 
        thinking,
        full_dataset, 
        exemplars=exemplars,
    ) -> list:
        if not thinking:
            max_tokens = 1000 
        elif 'opus' in model_id:
            max_tokens = 32000 // 4
        elif 'sonnet' in model_id:
            max_tokens = 64000 // 4
        else:
            max_tokens = 1000
        
        requests = [] 
        for split in ['train','test','validation']: 
            dataset = full_dataset[split]
            for item_idx, item in enumerate(dataset):
                custom_id = f"{split}-{item_idx}"
                
                message = []
                if fewshot:
                    for exemplar in exemplars:
                        message += [ 
                            {"role": "user", "content": format_question(exemplar)},
                            {"role": "assistant", "content": str(1 if exemplar['answer'] else 0)},  
                        ]
                message += [{"role": "user", "content": format_question(item)}]
                
                if thinking:
                    requests += [
                        Request(
                            custom_id=custom_id,
                            params=MessageCreateParamsNonStreaming(
                                model=model_id,
                                max_tokens=max_tokens,
                                system=system_prompt,
                                messages=message,
                                #temperature=0, 会报错
                                thinking={
                                    "type": "enabled",
                                    "budget_tokens": 2048
                                },
                            )
                        ),
                    ]
                else:
                    requests += [
                        Request(
                            custom_id=custom_id,
                            params=MessageCreateParamsNonStreaming(
                                model=model_id,
                                max_tokens=max_tokens,
                                system=system_prompt,
                                messages=message,
                                temperature=0,
                            )
                        ),
                    ]
        return requests


    def get_pred(item, thinking) -> str:
        '''Extract the prediction from a batch item'''
        text_idx = 1 if thinking else 0
        try:
            return item.result.message.content[text_idx].text
        except IndexError:
            return 'MissingTextBlock'
        
    models_cannot_think = [
        "claude-3-opus-20240229",
        "claude-3-5-haiku-20241022",
    ]
    full_dataset = datasets.load_dataset(
                dataset_name,
                token=TOKEN,
                revision=COMMIT,
    ) 
    
    if (model_id in models_cannot_think) and (thinking):
        raise ValueError(f"This is an invalid argument combination. {model_id} does not support extended thinking.")
    else:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        requests = compose_requests(model_id, fewshot, thinking, full_dataset=full_dataset)
        message_batch = client.messages.batches.create(requests=requests)
        jobid  = message_batch.id

    while True: # Wait for finishing
        batch = client.messages.batches.retrieve(jobid)
        if batch.processing_status != "in_progress":
            break
        time.sleep(10)  
    
        
    # Obtain the results, augment them with necessary information.
    instance_info = []
    for split in ['train','test','validation']:
        qa_dataset = full_dataset[split]
        qa_dataset['split'] = split
        instance_info.append(qa_dataset.to_pandas())
    instance_info = pd.concat(instance_info)
    instance_info["idx"] = instance_info.groupby("split").cumcount()

    outputs = client.messages.batches.results(jobid)
    predictions = [get_pred(item, thinking) for item in outputs]
    notanumbers = [(pred not in ['0','1'] ) for pred in predictions]
    binarised_predictions =[(int(pred) if pred in ['0','1'] else 0) for pred in predictions]
    
    df_out = pd.DataFrame({
        'model': model_id + '-reasoning' if thinking else model_id,
        'fewshot': fewshot,
        "custom_id": [item.custom_id for item in outputs],
        "prediction": binarised_predictions,
        "NotANumber": notanumbers,
    })
    df_out["split"] = df_out["custom_id"].apply(lambda x: x.split("-")[0])
    df_out["idx"]   = df_out["custom_id"].apply(lambda x: int(x.split("-")[1]))
    
    all_results = instance_info.merge(df_out, on=["split", "idx"])
    all_results['accuracy'] =  (all_results.prediction == all_results.answer)
    all_results['in_exemplars'] = all_results.id.isin(exemplar_ids)
    all_results = all_results[[
        'id', 'year', 'subject', 'subject_jp', 'theme', 'question_type',
        'instruction', 'question', 'label', 'answer', 'in_exemplars',
        'prediction', 'NotANumber', 'accuracy', 'split', 'model'
    ]].reset_index(drop=True)
    return all_results


def test_openai_model(model_id: str, fewshot: bool, dataset_name: str) -> pd.DataFrame:
    '''Evaluate an OpenAI model, organise the results as a Pandas DataFrame'''
    def is_reasoning_model(model_id: str) -> bool:
        return model_id.startswith(('o3', 'o4', 'gpt-5'))


    def create_and_save_batch(
        model_id: str, 
        fewshot: bool, 
        targeted_splits: list,
        full_dataset: datasets.arrow_dataset.Dataset, 
        exemplars : datasets.arrow_dataset.Dataset = exemplars
    ) -> pathlib.PosixPath:
        '''Create a jsonl file in order to use OpenAI batch API, returning the directory where the jsonl file is saved.'''
        if is_reasoning_model(model_id):
            endpoint = "/v1/responses"
            max_tokens = 128000 // 4 if 'gpt' in model_id else 100000 // 4
        else:
            endpoint = "/v1/chat/completions"
            max_tokens = 1000
        
        message_lst = []
        for split in targeted_splits:
            qa_dataset = full_dataset[split]
            for item_idx, item in enumerate(qa_dataset):
                custom_id = f"{split}-{item_idx}"
                
                if is_reasoning_model(model_id):
                    message = [
                        {
                            "role": "developer", 
                            "content": [{"type": "input_text", "text": system_prompt}]
                        }
                    ]
                    
                    if fewshot: 
                        for exemplar in exemplars:
                            message += [
                                    {
                                        "role": "user", 
                                        "content": [{"type": "input_text", "text": format_question(exemplar)}]
                                    },
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": str(1 if exemplar['answer'] else 0)}]
                                    },
                            ]
                    else:
                        pass
                    
                    message += [
                            {
                            "role": "user",
                            "content": [{"type": "input_text", "text": format_question(item)}]
                            }
                    ]
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": endpoint,
                        "body": {
                            "model": model_id, 
                            "input": message,
                            #"temperature": 0, 传入温度会报错
                            "max_output_tokens": max_tokens,
                            "text": {"format": {"type": "text"}},
                            "reasoning": {"effort": "medium"},
                        } 
                    }

                        
                else:
                    message = [{"role": "system", "content": system_prompt},]
                    if fewshot:           
                        for exemplar in exemplars:
                            message += [
                                {"role": "user", "content": exemplar['question']},
                                {"role": "assistant", "content": str(1 if exemplar['answer'] else 0)},
                            ]
                    else: #zeroshot
                        pass
                    
                    message += [{"role": "user", "content": item["question"]}]
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": endpoint,
                        "body": {
                            "model": model_id, 
                            "temperature": 0,
                            "messages": message,
                            "max_tokens": max_tokens
                            }
                    }
                    
                message_lst.append(request)
                
        os.makedirs(PROJECT_DIR / 'closed_model_dataset', exist_ok=True)
        save_dir = PROJECT_DIR / 'closed_model_dataset'/ f"{model_id}_{'fewshot' if fewshot else 'zeroshot'}.jsonl"
        with open(save_dir, "w", encoding="utf-8") as f:
            for item in message_lst:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")
                
        return save_dir
                
    def obtain_results(
        results: list, 
        model_id: str, 
        fewshot: bool, 
        instance_info: pd.DataFrame, 
        exemplar_ids: list = exemplar_ids 
    ) -> pd.DataFrame:
        '''Merge the prediction results with instance_info as the final outputs.'''
        if is_reasoning_model(model_id):
            predictions = [item['response']['body']['output'][-1]['content'][0]['text'] for item in results]
        else:
            predictions = [item['response']['body']['choices'][0]['message']['content'] for item in results]
            
        notanumbers = [ (pred not in ['0','1'] ) for pred in predictions]
        predictions =[ (int(pred) if pred in ['0','1'] else 0) for pred in predictions]
        splits = [item['custom_id'].split('-')[0] for item in results]
        indices = [item['custom_id'].split('-')[1] for item in results]
        df = pd.DataFrame({
            'model': model_id,
            'fewshot': fewshot,
            'prediction': predictions,
            'NotANumber': notanumbers,
            'split': splits,
            'idx': indices
        })
        assert len(instance_info) == len(df)
        df = pd.concat([instance_info.reset_index(drop=True),df], axis=1)
        df['accuracy'] =  (df.prediction == df.answer)
        df['in_exemplars'] = df.id.isin(exemplar_ids)
        
        return df[[
            'id', 'year', 'subject', 'subject_jp', 'theme', 'question_type',
            'instruction', 'question', 'label', 'answer', 'in_exemplars',
            'prediction', 'NotANumber', 'accuracy', 'split', 'model'
            ]].reset_index(drop=True)
        
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    full_dataset = datasets.load_dataset(
        dataset_name,
        token=TOKEN,
        revision=COMMIT,
    ) 
    targeted_splits = ["train","test","validation"]

    endpoint = "/v1/responses" if is_reasoning_model(model_id) else "/v1/chat/completions"
            
    file_name = create_and_save_batch(model_id, fewshot, targeted_splits=targeted_splits, full_dataset=full_dataset)
    batch_input_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={
            "description": "eval on bar exam",
            "model_id": model_id,
            "fewshot": fewshot,
        }
    )
    batch_id = batch.id
    
    while True:
        batch_status = client.batches.retrieve(batch_id).status
        if batch_status == "completed":
            break
        else:
            time.sleep(10)

    instance_info = []
    for split in targeted_splits:
        qa_dataset = full_dataset[split]
        instance_info.append(qa_dataset.to_pandas())
    instance_info = pd.concat(instance_info)

    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id
    results = [json.loads(line) for line in client.files.content(output_file_id).text.splitlines()]
    return obtain_results(results, model_id, fewshot, instance_info=instance_info)