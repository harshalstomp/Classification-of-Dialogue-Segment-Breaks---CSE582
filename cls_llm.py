import argparse
import random
import numpy as np
import pandas as pd
import torch
import ast
import csv
from tqdm import tqdm

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# suggestion: use mistral or llama
MODELS = ['stablelm', 'mistral', 'llama']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_access_token = ''


def create_sys_prompt(model):
    if model == 'stablelm':
        sys_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
                    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
                    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
                    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
                    - StableLM will refuse to participate in anything that could harm a human.
                    """
    else:
        sys_prompt = ''
    return sys_prompt

def create_prompt_zero(parsed_line, model, sys_prompt):
    user1 = parsed_line['user1']
    user2 = parsed_line['user2']
    utter1 = parsed_line['utter1']
    utter2 = parsed_line['utter2']
    intent1 = parsed_line['intent1']
    intent2 = parsed_line['intent2']
    if model == 'stablelm':
        prompt = """{SYSTEM_PROMPT}
                    <|USER|>
                    The following pair of utterances is extracted from a long dialogue between a `human` and an `alien`.
                    The `intent` label indicates the intent of each utterance.
                    Please answer if they belong to the same dialogue topic segment.
                    There are 2 cases: `yes` if there is no break in the activity between the two utterances, `no` if there is a shift where Utterance#1 occurs at the end of a segment and Utterance#2 begins a new segment.
                    Your answer should be either `yes` or `no` without any additional information or explanation.
                    Utterance#1 - {user1}: {utter1} (intent: {intent1})
                    Utterance#2 - {user2}: {utter2} (intent: {intent2})
                    <|ASSISTANT|>""".format(system_prompt=sys_prompt, user1=user1, user2=user2, utter1=utter1, utter2=utter2, intent1=intent1, intent2=intent2)
        return prompt
    
    if model == 'mistral' or model == 'llama':
        prompt = """<s>[INST] <<SYS>>
                    You are an expert in analysing dialogues.
                    <</SYS>>
                    The following pair of utterances is extracted from a long dialogue between a `human` and an `alien`.
                    Attached `intent` labels indicate the underlying intent of each utterance.
                    Please answer if they belong to the same dialogue topic segment.
                    Your answer should be either `yes` or `no` without any additional information or explanation.
                    Utterance 1 - {user1}: {utter1} (intent: {intent1})
                    Utterance 2 - {user2}: {utter2} (intent: {intent2}) [/INST]""".format(user1=user1, user2=user2, utter1=utter1, utter2=utter2, intent1=intent1, intent2=intent2)
        return prompt

def create_prompt_few(parsed_test, parsed_trains, model):
    num_shot = len(parsed_trains)
    shot_prompts = []
    shot_qs = []
    shot_as = []
    for i in range(num_shot):
        user1 = parsed_trains[i]['user1']
        user2 = parsed_trains[i]['user2']
        utter1 = parsed_trains[i]['utter1']
        utter2 = parsed_trains[i]['utter2']
        intent1 = parsed_trains[i]['intent1']
        intent2 = parsed_trains[i]['intent2']
        label = parsed_trains[i]['label']
        label = 'yes' if label == 0 else 'no'
        shot_prompt = """Utterance#1 - {user1}: {utter1}
                         Utterance#2 - {user2}: {utter2}
                         Answer: {label}""".format(user1=user1, user2=user2, utter1=utter1, utter2=utter2, label=label)
        shot_prompts.append(shot_prompt)
        shot_q = """Utterance 1 - {user1}: {utter1}
                    Utterance 2 - {user2}: {utter2}""".format(user1=user1, user2=user2, utter1=utter1, utter2=utter2)
        shot_a = label
        shot_qs.append(shot_q)
        shot_as.append(shot_a)
    shot_prompts = '\n'.join(shot_prompts)
    user1, user2, utter1, utter2, intent1, intent2 = parsed_test['user1'], parsed_test['user2'], parsed_test['utter1'], parsed_test['utter2'], parsed_test['intent1'], parsed_test['intent2']
    if model == 'stablelm':
        prompt = """<|SYSTEM|>
                    You are an expert in analyzing dialogues.
                    <|USER|>
                    The following pair of utterances is extracted from a long dialogue between a `human` and an `alien`.
                    Please answer if they belong to the same dialogue topic segment.
                    Your answer should be either `yes` or `no` without any additional information or explanation.
                    Refer to the following examples for the context:
                    {shot_prompts}
                    Test Example:
                    Utterance#1 - {user1}: {utter1} (intent: {intent1})
                    Utterance#2 - {user2}: {utter2} (intent: {intent2})
                    Answer: 
                    <|ASSISTANT|>""".format(shot_prompts=shot_prompts, user1=user1, user2=user2, utter1=utter1, utter2=utter2, intent1=intent1, intent2=intent2)
    
    elif model == 'mistral' or model == 'llama':
        prompt = """<s>[INST] <<SYS>>
                    You are an expert in analysing dialogues.
                    The following pair of utterances is extracted from a long dialogue between a `human` and an `alien`.
                    Please answer if they belong to the same dialogue topic segment.
                    Your answer should be either `yes` or `no` without any additional information or explanation.
                    <</SYS>>
                    """
        for i in range(num_shot):
            prompt += """{shot_q} [/INST]
                         {shot_a}""".format(shot_q=shot_qs[i], shot_a=shot_as[i])
            prompt += """</s><s>[INST]"""
        prompt += """Utterance 1 - {user1}: {utter1}
                     Utterance 2 - {user2}: {utter2} [/INST]""".format(user1=user1, user2=user2, utter1=utter1, utter2=utter2)
    
    return prompt

def select_few_shot(train_data, k, parsed_test):
    test_int_1 = parsed_test['intent1']
    test_int_2 = parsed_test['intent2']
    parsed_trains = []
    for i in range(train_data.shape[0]):
        line = train_data.iloc[i]
        parsed_line = parse_line(line, 'train')
        train_int_1 = parsed_line['intent1']
        train_int_2 = parsed_line['intent2']
        if train_int_1 == test_int_1 and train_int_2 == test_int_2:
            parsed_trains.append(parsed_line)
        if len(parsed_trains) == k:
            break
    if len(parsed_trains) < k:
        idxs = random.sample(range(train_data.shape[0]), k - len(parsed_trains))
        for idx in idxs:
            line = train_data.iloc[idx]
            parsed_line = parse_line(line, 'train')
            parsed_trains.append(parsed_line)

    return parsed_trains

def get_response(prompt, model, tokenizer):
    # input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=512)
    n = input_ids.input_ids.shape[1]
    new_tokens = outputs[0][n:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def parse_file(file_path):
    data = pd.read_csv(file_path)
    return data

def parse_line(line, split):
    turn1 = line['utterance1']
    turn2 = line['utterance2']
    turn1 = ast.literal_eval(turn1)
    turn2 = ast.literal_eval(turn2)

    user1 = turn1['user']
    user2 = turn2['user']
    utter1 = turn1['text']
    utter2 = turn2['text']
    intent1 = turn1['intent']
    intent2 = turn2['intent']

    label = 'yes' if line['label'] == 0 else 'no'

    if split == 'train':
        return {'user1': user1, 'user2': user2, 'utter1': utter1, 'utter2': utter2, 'intent1': intent1, 'intent2':intent2, 'label': label}
    elif split == 'test':
        return {'user1': user1, 'user2': user2, 'utter1': utter1, 'utter2': utter2, 'intent1': intent1, 'intent2':intent2}
    else:
        raise ValueError('Invalid split')

def model_path(model):
    if model == 'stablelm':
        path = "stabilityai/stablelm-tuned-alpha-7b"
    elif model == 'mistral':
        path = "mistralai/Mistral-7B-Instruct-v0.1"
    elif model == 'llama':
        path = "meta-llama/Llama-2-13b-chat-hf"
    else:
        raise ValueError('Invalid model')
    return path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='train.csv', help='Path to training file')
    parser.add_argument('--test_path', type=str, default='test.csv', help='Path to test file')
    parser.add_argument('--output_path', type=str, default='results/output_llama.csv', help='Path to output file')
    parser.add_argument('--model', type=str, default='mistral', help='Model to use for generation')
    parser.add_argument('--evaluate', type=bool, default=False, help='Compute evaluation metrics based on output file')
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot learning mode')
    parser.add_argument('--k', type=int, default=2, help='The number of examples to use in few shot learning mode')
    parser.add_argument('--seed', type=int, default=0, help='Global seed for reproducibility')
    parser.add_argument('--hf_access_token', type=str, default='', help='Hugging Face access token')
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model
    assert model_name in MODELS, f'Invalid model: {model_name}'
    path = model_path(model_name)

    if model_name == 'stablelm' or model_name == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', trust_remote_code=True) # torch_dtype=torch.float16
    elif model_name == 'llama':
        hf_access_token = args.hf_access_token
        tokenizer = AutoTokenizer.from_pretrained(path, token=hf_access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', token=hf_access_token)

    test_data = parse_file(args.test_path)
    k = args.k
    if args.few_shot:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_data = parse_file(args.train_path)
        for idx in tqdm(range(test_data.shape[0])):
            line = test_data.iloc[idx]
            parsed_test = parse_line(line, 'test')
            
            parsed_trains = select_few_shot(train_data, k, parsed_test)

            prompt = create_prompt_few(parsed_test, parsed_trains, model_name)
            response = get_response(prompt, model, tokenizer)
            response = response.strip()
            with open(args.output_path, 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([response])
    else:
        for idx in tqdm(range(test_data.shape[0])):
            line = test_data.iloc[idx]
            parsed_test = parse_line(line, split='test')
            prompt = create_prompt_zero(parsed_test, model_name, create_sys_prompt(model))
            response = get_response(prompt, model, tokenizer)
            response = response.strip()

            with open(args.output_path, 'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([response])

    if args.evaluate:
        gts = list(test_data['label'].values)
        outputs = pd.read_csv(args.output_path, header=None)
        preds = []
        for idx in range(outputs.shape[0]):
            response = outputs.iloc[idx][0]
            if 'yes' in response.lower():
                prediction = 0
            elif 'no' in response.lower():
                prediction = 1
            else:
                raise ValueError('Invalid response')
            preds.append(prediction)
        print(classification_report(gts, preds))
        
        
if __name__ == '__main__':
    main()