import torch
import math
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt-neo-2.7B')
model = AutoModelForCausalLM.from_pretrained('gpt-neo-2.7B')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

def calculate_ppls(sentences, device):
    print('calculate PPL scores...')
    ppls = []
    for s_list in tqdm(sentences):
        cur_list = []
        for r in s_list:
            inputs = tokenizer(r, return_tensors='pt', truncation=True, max_length=500).data
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            output = model(**inputs, labels=inputs['input_ids'])
            loss = output[0]

            # testing
            logits = output.logits
            labels=inputs['input_ids']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print ('loss1 = ', loss1)

            cur_list.append(min(math.exp(loss.item()), 200)) # sentence-level PPL
        ppls.append(sum(cur_list) / len(cur_list)) # example-level PPL
    return ppls

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps

def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_path", type=str, default='./emnlp_data/nq/random_testset/nq_ref')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    hyps = read_hyp(args.hyp_path)
    sentences = []
    for hyp in hyps:
        hyp_sents = sent_tokenize(hyp)
        sentences.append(hyp_sents)
    assert len(sentences) == 500, len(sentences)
    ppls = calculate_ppls(sentences, device)
    assert len(ppls) == 500, len(ppls)

    inverse_ppls = [1 / p for p in ppls]
    coh_sent = np.nanmean(inverse_ppls)

    with open(args.hyp_path + '_avg_sent_ppl', 'w') as outfile:
        json.dump(ppls, outfile)
        outfile.write('\n')
        json.dump(coh_sent, outfile)
