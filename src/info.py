import torch
import math
from tqdm import tqdm
import numpy as np
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt-neo-2.7B')
model = AutoModelForCausalLM.from_pretrained('gpt-neo-2.7B').half()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

def read_testfile(testfile):
    '''read testset from wow'''
    res = []
    with open(testfile, 'r', encoding='utf-8') as r:
        for i, line in enumerate(r):
            parts = line.strip().split('\t')
            assert len(parts) == 4, parts
            res.append(parts)
    return res

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps

def calculate_info_per_example(hyps_knowledge, queries, topics, args):
    info_seq = []
    for hyp, query, topic in tqdm(zip(hyps_knowledge, queries, topics)):
        hyp = ' '.join(hyp.split()[:300])
        if args.task == 'nq' and query.strip()[-1] != '?':
            query = query.strip() + '?'
        instruction = f"Generate a Wikipedia to answer the given question.\nTopic: {topic.strip()}.\nQuestion: {query.strip()}\nWikipedia: "
        example = instruction + hyp
        inputs = tokenizer(example, return_tensors='pt', truncation=True).data
        prefix = tokenizer(instruction, return_tensors='pt', truncation=True).data
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        for k, v in prefix.items():
            prefix[k] = v.to(device)
        output = model(**inputs, labels=inputs['input_ids'])
        logits = output.logits
        labels=inputs['input_ids']
        logits = logits[:, prefix['input_ids'].shape[-1]:, :]
        labels = labels[:, prefix['input_ids'].shape[-1]:]
        assert logits.shape[1] == labels.shape[1], (logits.shape, labels.shape)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        info = 1 - torch.exp(-loss1)
        info = info.item()
        info_seq.append(info)

    return info_seq

def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='nq')
    parser.add_argument("--hyp_path", type=str)
    parser.add_argument("--ref_path", type=str, default='./emnlp_data/nq/random_testset/nq_test_random_testset.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    hyps = read_hyp(args.hyp_path)
    testset = read_testfile(args.ref_path)
    queries = [t[1].strip() for t in testset]
    topics = [t[0].strip() for t in testset]
    assert len(hyps) == len(testset) == len(queries) == 500, (len(hyps), len(testset))

    info_list = calculate_info_per_example(hyps, queries, topics, args)
    assert len(info_list) == 500, len(info_list)

    print ('mean info = ', np.nanmean(info_list))

    with open(args.hyp_path + '_info', 'w') as outfile:
        for info in info_list:
            outfile.write(str(info) + '\n')