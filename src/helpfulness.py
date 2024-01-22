import torch
import tqdm
import math
import numpy as np
import argparse
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


def read_testfile(testfile):
    '''read testset (groud truth knowledge, response)'''
    res = []
    with open(testfile, 'r', encoding='utf-8') as r:
        for i, line in enumerate(r):
            parts = line.strip().split('\t')
            assert len(parts) == 4, parts
            res.append(parts)
    # topic, query, knowledge, response
    return res

def read_knowledge_prompt(prompt_file):
    '''
    prompt_file:
    {last_utter: ["(last_utter) topic => knowledge", "..", ..]}
    '''
    knowledge_prompts = []
    with open(prompt_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line_list = eval(line)[:8]
            knowledge_prompts.append(line_list)        
    return knowledge_prompts

def read_hyp_knowledge(path):
    '''read generated knowledge (by dpr or llm)'''
    with open(path, 'r', encoding='utf-8') as r:
        res = [line.strip() for line in r]
    assert len(res) == 500, len(res)
    return res

def load_model(model_name):
    if 'flan' in model_name:
        # flan-t5-xxl: 11B = 22G, takes 6 min to load model. 1 min if gpus are empty.
        assert model_name in ['flan-t5-xxl', 'flan-t5-xl', 'flan-t5-large', 'flan-t5-base', 'flan-t5-small']
        tokenizer = T5Tokenizer.from_pretrained(f"google/{model_name}", local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(f"google/{model_name}", device_map="balanced_low_0", torch_dtype=torch.float16, local_files_only=True)
    elif 'llama' in model_name:
        path = '/apdcephfs/share_1594716/chenliang/cache/llama1/65B'
        tokenizer = LlamaTokenizer.from_pretrained(path, padding_side='left') # left-padding for decoder-only model
        tokenizer.pad_token, tokenizer.bos_id, tokenizer.eos_id = -1, 1, 2
        model = LlamaForCausalLM.from_pretrained(path, device_map="balanced_low_0", torch_dtype=torch.float16)
    else:
        ''' A decoder-only architecture is being used, but right-padding was detected!
        For correct generation results, please set `padding_side='left'` when initializing the tokenizer.'''
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}", use_fast=False, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(f"facebook/{model_name}", device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def compute_ppl(prefix_and_output_text=None, output_text=None, model=None, tokenizer=None, infer_gpu=0):
    '''calculate ppl for a single response'''
    with torch.no_grad():
        tokd_inputs = tokenizer.encode(prefix_and_output_text, return_tensors="pt")
        tokd_inputs = tokd_inputs.to(infer_gpu)

        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenizer.encode(output_text, return_tensors="pt")

        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:, :tokd_labels.shape[1] - tokd_suffix.shape[1] + 1] = -100 # mask out the prefix

        outputs = model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--task', type=str, default='nq')

    parser.add_argument("--debug", type=boolean_string, default=True)
    parser.add_argument("--zero_shot", type=boolean_string, default=False)

    parser.add_argument('--testfile', type=str, default='data/testset.txt')
    parser.add_argument('--promptfile', type=str, default='data/testset.txt')
    parser.add_argument('--hyp_knowledge', type=str, default='')

    parser.add_argument('--downstream_model', type=str)
    parser.add_argument("--knowledge_type", type=str, default='wo_knowledge', help='wo_knowledge, w_ref_knowledge, w_hyp_knowledge, random_knowledge')

    parser.add_argument('--infer_gpu', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    testset = read_testfile(args.testfile)
    nq_prompt_list = read_knowledge_prompt(args.promptfile)
    random_knowledge_list = [random.choice(nq_prompt_list[499 - i]).split('\t')[-2] for i in range(len(nq_prompt_list))]

    if args.hyp_knowledge:
        hyp_knowledge_list = read_hyp_knowledge(args.hyp_knowledge)
        assert len(hyp_knowledge_list) == len(testset), len(hyp_knowledge_list)

    if args.debug:
        testset = testset[:3]

    tokenizer, model = load_model(args.downstream_model)

    loss_list, ppl_list = [], []
    for i in tqdm.tqdm(range(len(testset))):
        topic, query, knowledge, response = testset[i]
        examples = [e.split('\t') for e in nq_prompt_list[i] if len(e.split('\t')) == 4]
        turns = query.split(" [SEP] ")
        last_turn = turns[-1].strip()

        ref_knowledge = knowledge.strip()
        truncate_len = 500
        if len(ref_knowledge.split(' ')) > truncate_len:
            print (f'Warning: knowledge {i} length {len(ref_knowledge.split(" "))} exceeds {truncate_len}, truncating to {truncate_len}')
            ref_knowledge = ' '.join(ref_knowledge.split(' ')[:truncate_len]).strip()
        if args.hyp_knowledge:
            hyp_knowledge = hyp_knowledge_list[i]

        infer_sample = f"Passage:\nQuery: {last_turn.strip()}\nAnswer: " # set to empty passage
        if args.knowledge_type == 'w_hyp_knowledge':
            infer_sample = f"Passage: {hyp_knowledge.strip()}\nQuery: {last_turn.strip()}\nAnswer: "
        elif args.knowledge_type == 'w_ref_knowledge':
            infer_sample = f"Passage: {ref_knowledge}\nQuery: {last_turn.strip()}\nAnswer: "
        elif args.knowledge_type == 'random_knowledge':
            infer_sample = f"Passage: {random_knowledge_list[i].strip()}\nQuery: {last_turn.strip()}\nAnswer: "

        prompt = ''
        cur_len = 0
        if args.zero_shot:
            if args.knowledge_type == 'wo_knowledge':
                if args.task == 'nq':
                    prompt = f'Read the passage and answer the question below:\nPassage: {ref_knowledge}\nQuestion: {last_turn}\nAnswer: '
                elif args.task == 'wow':
                    prompt = f'Using the knowledge from the passage, complete the dialogue below:\nPassage: {ref_knowledge}\nSpeaker 1: {last_turn}\nSpeaker 2: '

            elif args.knowledge_type == 'w_ref_knowledge':
                if args.task == 'nq':
                    prompt = f'Read the passage and answer the question below:\nPassage: {ref_knowledge}\nQuestion: {last_turn}\nAnswer: '
                elif args.task == 'wow':
                    prompt = f'Using the knowledge from the passage, complete the dialogue below:\nPassage: {ref_knowledge}\nSpeaker 1: {last_turn}\nSpeaker 2: '
            elif args.knowledge_type == 'w_hyp_knowledge':
                if args.task == 'nq':
                    prompt = f'Read the passage and answer the question below:\nPassage: {hyp_knowledge}\nQuestion: {last_turn}\nAnswer: '
                elif args.task == 'wow':
                    prompt = f'Using the knowledge from the passage, complete the dialogue below:\nPassage: {hyp_knowledge}\nSpeaker 1: {last_turn}\nSpeaker 2: '
            else:
                raise NotImplementedError(args.knowledge_type)
        else:
            for example in examples:
                p_topic, p_turns, p_knowledge, p_response = [e.strip() for e in example]
                if p_knowledge.startswith(p_topic):
                    p_knowledge = p_knowledge[len(p_topic):]

                demonstration = f"Passage: {p_knowledge.strip()}\nQuery: {p_turns.split(' [SEP] ')[-1].strip()}\nAnswer: {p_response.strip()}"
                
                if cur_len < 1800 - len(infer_sample.split(' ')):
                    prompt += demonstration + '\n\n'
                    cur_len += len(demonstration.split(' '))

            prompt += infer_sample

        prefix_and_output_text = prompt + response
        output_text = response
        loss, ppl = compute_ppl(prefix_and_output_text, output_text, model, tokenizer, args.infer_gpu)
        loss_list.append(loss)
        ppl_list.append(ppl)

        if args.debug:
            print (prefix_and_output_text)
            print (loss, ppl)

    with open(f'helpfulness_results/{args.exp_name}.txt', 'w') as f:
        f.write(str(loss_list).strip() + '\n')
        f.write(str(ppl_list).strip() + '\n')

        f.write(f'loss: {np.mean(loss_list)}\t{np.std(loss_list)}\t{np.var(loss_list)}\n')
        f.write(f'ppl: {np.mean(ppl_list)}\t{np.std(ppl_list)}\t{np.var(ppl_list)}\n')
