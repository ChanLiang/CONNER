from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import Counter
import copy
import json
import argparse
import random
random.seed(42)

import numpy as np
from factuality_metric import ner_metric, nli_metric_batch
from src.claim_handling import obtain_important_ne
from tools import WikiSearch

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps

def read_IR_docs(IR_path):
    IR_docs = []
    with open(IR_path, 'r') as infile:
        for line in infile:
            IR_docs.append(json.loads(line.strip()))
    return IR_docs

def read_testfile(testfile):
    '''read testset from wow'''
    res = []
    with open(testfile, 'r', encoding='utf-8') as r:
        for i, line in enumerate(r):
            parts = line.strip().split('\t')
            assert len(parts) == 4, parts
            res.append(parts)
    # topic, query, knowledge, response
    return res

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--hyp_path', type=str, default=None, help='path to generations to evaluate') 
    parser.add_argument('--ref_path', type=str, default=None, help='path to generations to evaluate') 
    parser.add_argument('--eval_num', type=int, default=-1)

    parser.add_argument('--use_IR_eval', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 
    parser.add_argument('--retrieved_num', type=int, default=3)
    parser.add_argument('--wo_ground_truth_knowledge', type=boolean_string, default='False')

    parser.add_argument('--debug', type=boolean_string) 
    parser.add_argument('--save_gen_for_analysis', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 

    args = parser.parse_args()
    return args

def single_instance_eval(hyp, response, recall_list, args):
    # multiple evidences
    nli_contradict_prob, nli_entail_prob, nli_neutral_prob, nli_label = [], [], [], []

    if args.use_IR_eval and args.retrieved_num:
        assert recall_list and len(recall_list) >= 10, f"len(recall_list) = {len(recall_list)}"
        recall_list = recall_list[:args.retrieved_num]

    # NLI: identify the evs that give highest nli entailment score
    premise_hypothesis_pairs = [[ev, hyp] for ev in [response] + recall_list]
    if len(premise_hypothesis_pairs) > 32:
        premise_hypothesis_pairs = premise_hypothesis_pairs[:32]
    bz = 8
    nli_probs, labels = [], []
    for t in range((len(premise_hypothesis_pairs) - 1) // bz + 1):
        bz_nli_probs, bz_labels = nli_metric_batch(premise_hypothesis_pairs[t * bz: min((t + 1) * bz, len(premise_hypothesis_pairs))])
        nli_probs.extend(bz_nli_probs)
        labels.extend(bz_labels)
    assert len(nli_probs) == len(premise_hypothesis_pairs) == len(labels), f"len(nli_probs) = {len(nli_probs)}, len(premise_hypothesis_pairs) = {len(premise_hypothesis_pairs)}, len(labels) = {len(labels)}"
    
    # [contradiction, neutral, entailment]
    entailment_argmax = np.argmax([nli_s[2] for nli_s in nli_probs])
    max_prob = nli_probs[entailment_argmax]
    max_label = labels[entailment_argmax]

    nli_contradict_prob.append(max_prob[0])
    nli_neutral_prob.append(max_prob[1])
    nli_entail_prob.append(max_prob[2])

    nli_label.append(max_label)
    # print (max_label, premise_hypothesis_pairs[entailment_argmax])

    idx = nli_label.index(max(nli_label))
    nli_label = max(nli_label)
    nli_contradict_prob = nli_contradict_prob[idx]
    nli_neutral_prob = nli_neutral_prob[idx]
    nli_entail_prob = nli_entail_prob[idx]

    eval_result_obj = {
        'premise_hypothesis_pairs': premise_hypothesis_pairs,
        'nli-label': nli_label,
        'nli-contr': nli_contradict_prob,
        'nli-entail': nli_entail_prob,
        'nli-neutr': nli_neutral_prob
    }

    return eval_result_obj

def main(args):

    # read hyp, ref, IR_docs
    hyps = read_hyp(args.hyp_path) 
    IR_recalls = read_IR_docs(args.hyp_path + '_IR_docs') 
    testset = read_testfile(args.ref_path)
    assert len(hyps) == len(testset) == len(IR_recalls) == 500, (len(hyps), len(testset), len(IR_recalls))

    # DEBUG mode!
    if args.debug:
        DEBUG_SAMPLE_SIZE = 10
        hyps = hyps[:DEBUG_SAMPLE_SIZE]
        IR_recalls = IR_recalls[:DEBUG_SAMPLE_SIZE]
        testset = testset[:DEBUG_SAMPLE_SIZE]
    
    final_contradict_prob, final_neutral_prob, final_entail_prob, all_nli_labels = [], [], [], []
    all_analysis_list = []

    for i in tqdm(range(len(hyps))):
        hyp, example, recall_list = hyps[i], testset[i], IR_recalls[i]
        response = example[3]

        res_obj = single_instance_eval(hyp, response, recall_list, args)
        if args.debug:
            print ('==' * 20)
            print (res_obj)

        final_contradict_prob.append(res_obj['nli-contr'])
        final_neutral_prob.append(res_obj['nli-neutr'])
        final_entail_prob.append(res_obj['nli-entail'])
        all_nli_labels.append(res_obj['nli-label'])
        all_analysis_list.append(res_obj)

    # analysis
    avg_contradict_prob = np.mean(final_contradict_prob)
    avg_neutral_prob = np.mean(final_neutral_prob)
    avg_entail_prob = np.mean(final_entail_prob)

    print("AVG PROBS: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(avg_contradict_prob*100, avg_neutral_prob*100, avg_entail_prob*100))

    nli_contradict_class_ratio, nli_neutral_class_ratio, nli_entail_class_ratio = 0, 0, 0

    nli_counter = Counter(all_nli_labels)

    nli_contradict_class_ratio=nli_counter[0]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    nli_neutral_class_ratio=nli_counter[1]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    nli_entail_class_ratio=nli_counter[2]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    
    print("NLI CLASS %: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(
        nli_contradict_class_ratio*100,
        nli_neutral_class_ratio*100,
        nli_entail_class_ratio*100
    ))

    res_path = args.hyp_path + '_factuality_results.txt'
    with open(res_path, 'a') as outfile:
        res_obj = {
            'Contradict_probs': avg_contradict_prob, 
            'Neutral_probs': avg_neutral_prob,
            'Entail_probs': avg_entail_prob,
            "nli_contradict_class_ratio": nli_contradict_class_ratio,
            "nli_neutral_class_ratio": nli_neutral_class_ratio, 
            "nli_entail_class_ratio": nli_entail_class_ratio,
        }
        json.dump(res_obj, outfile)
        outfile.write("\n")

    ana_path = args.hyp_path + '_analysis.txt'
    with open(ana_path, 'a') as outfile:
        json.dump(all_analysis_list, outfile)
        outfile.write("\n")


if __name__ == '__main__':
    args = args_parser()
    main(args)
