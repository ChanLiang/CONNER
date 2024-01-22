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

def read_ref(ref_path):
    doc_refs = []
    if 'json' not in ref_path: # txt: for wow
        with open(ref_path, 'r') as infile:
            for line in infile:
                parts = line.strip().split('\t')
                # topic, query, knowledge, response
                assert len(parts) == 4, parts
                doc_refs.append(parts[2])
    else: # json: for QA dataset
        with open(ref_path, 'r') as infile:
            data_list = json.load(infile)['data']
            for data in data_list:
                doc_refs.append(data['context'])
    return doc_refs

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--hyp_path', type=str, default=None, help='path to generations to evaluate') 
    parser.add_argument('--ref_path', type=str, default=None, help='path to generations to evaluate') 
    parser.add_argument('--eval_num', type=int, default=-1)
    parser.add_argument('--outer_strategy', type=str, default='max', help='max, min, mean') 

    parser.add_argument('--use_IR_eval', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 
    parser.add_argument('--retrieved_num', type=int, default=3)
    parser.add_argument('--wo_ground_truth_knowledge', type=boolean_string, default='True')

    parser.add_argument('--debug', type=boolean_string) 
    parser.add_argument('--save_gen_for_analysis', action='store_true', help='Flag for saving some lm-gens with its metric for analysis') 

    args = parser.parse_args()
    return args

def single_instance_eval(hyp, doc_ref_str, recall_list_, args):
    # multiple evidences
    hallu_ner_ratio = []
    nli_contradict_prob, nli_entail_prob, nli_neutral_prob, nli_label = [], [], [], []

    hyp_sents = sent_tokenize(hyp)
    doc_ref = sent_tokenize(doc_ref_str) + [doc_ref_str] if doc_ref_str else [] 

    retrieve_error = ''
    for sent in hyp_sents: 
        cur_doc_ref = copy.deepcopy(doc_ref) #
        recall_list = copy.deepcopy(recall_list_)

        if args.use_IR_eval and args.retrieved_num:
            assert recall_list and len(recall_list) >= 10, f"len(recall_list) = {len(recall_list)}"
            try:
                if not recall_list: # dont do this
                    recall_list = WikiSearch(sent, args.retrieved_num) # already sentences # raise ConnectTimeout(e, request=request)
                    assert len(recall_list) == args.retrieved_num, f"len(recall_list) = {len(recall_list)}, args.retrieved_num = {args.retrieved_num}"
                else:
                    recall_list = recall_list[:args.retrieved_num]
            except: # need to log
                retrieve_error = f"!!!error!!!:{sent}"
        else:
            recall_list = []

        # 1. NER
        sent_obj_with_ne = obtain_important_ne(sent.strip())
        NE_to_check = sent_obj_with_ne['important_ne'] + sent_obj_with_ne['unimportant_ne']
        if NE_to_check:
            correct_ner_ratio = 0
            if not args.wo_ground_truth_knowledge: # ref
                correct_ner_ratio = ner_metric(NE_to_check, doc_ref_str) # apply directly on wiki and/or google search snippets
            for recall_passage in recall_list:
                correct_ner_ratio = max(correct_ner_ratio, ner_metric(NE_to_check, recall_passage))
            hallu_ner_ratio.append(1 - correct_ner_ratio)

        # 2. NLI: identify the evs that give highest nli entailment score
        premise_hypothesis_pairs = [[ev, sent] for ev in cur_doc_ref + recall_list]
        if args.wo_ground_truth_knowledge:
            premise_hypothesis_pairs = [[ev, sent] for ev in recall_list]
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

    hallu_ner_ratio = np.nanmean(hallu_ner_ratio)
    idx = None
    if args.outer_strategy == 'max':
        idx = nli_label.index(max(nli_label))
        nli_label = max(nli_label)
    if args.outer_strategy == 'min':
        idx = nli_label.index(min(nli_label))
        nli_label = min(nli_label)

    if args.outer_strategy != 'mean':
        nli_contradict_prob = nli_contradict_prob[idx]
        nli_neutral_prob = nli_neutral_prob[idx]
        nli_entail_prob = nli_entail_prob[idx]
    else: # mean
        nli_contradict_prob = np.nanmean(nli_contradict_prob)
        nli_neutral_prob = np.nanmean(nli_neutral_prob)
        nli_entail_prob = np.nanmean(nli_entail_prob)

    eval_result_obj = {
        'claim_to_verify': hyp_sents,
        'doc_ref': doc_ref,
        'recall_list': recall_list,
        'retrieve_error': retrieve_error,

        'hallu_ner': hallu_ner_ratio,
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
    doc_refs = read_ref(args.ref_path) # txt file
    assert len(hyps) == len(doc_refs) == len(IR_recalls) == 500, (len(hyps), len(doc_refs), len(IR_recalls))

    # DEBUG mode!
    if args.debug:
        DEBUG_SAMPLE_SIZE = 5
        hyps = hyps[:DEBUG_SAMPLE_SIZE]
        IR_recalls = IR_recalls[:DEBUG_SAMPLE_SIZE]
        doc_refs = doc_refs[:DEBUG_SAMPLE_SIZE]
    
    final_hallu_ner_score = []
    final_contradict_prob, final_neutral_prob, final_entail_prob, all_nli_labels = [], [], [], []
    all_analysis_list = []

    for i in tqdm(range(len(hyps))):
        hyp, doc_ref, recall_list = hyps[i], doc_refs[i], IR_recalls[i]

        res_obj = single_instance_eval(hyp, doc_ref, recall_list, args)

        final_hallu_ner_score.append(res_obj['hallu_ner'])
        final_contradict_prob.append(res_obj['nli-contr'])
        final_neutral_prob.append(res_obj['nli-neutr'])
        final_entail_prob.append(res_obj['nli-entail'])
        all_nli_labels.append(res_obj['nli-label'])
        all_analysis_list.append(res_obj)

    # analysis
    avg_hallu_ner_ratio = np.nanmean(final_hallu_ner_score)
    avg_contradict_prob = np.mean(final_contradict_prob)
    avg_neutral_prob = np.mean(final_neutral_prob)
    avg_entail_prob = np.mean(final_entail_prob)

    print("\nHallu NER: {:.2f}%".format(avg_hallu_ner_ratio*100))
    print("AVG PROBS: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(avg_contradict_prob*100, avg_neutral_prob*100, avg_entail_prob*100))

    nli_contradict_class_ratio, nli_neutral_class_ratio, nli_entail_class_ratio = 0, 0, 0

    if args.outer_strategy == 'mean':
        all_nli_labels = [item for sublist in all_nli_labels for item in sublist]
    nli_counter = Counter(all_nli_labels)

    nli_contradict_class_ratio=nli_counter[0]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    nli_neutral_class_ratio=nli_counter[1]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    nli_entail_class_ratio=nli_counter[2]/(nli_counter[0]+nli_counter[1]+nli_counter[2])
    
    print("NLI CLASS %: Contradict: {:.2f}%, Neutral: {:.2f}%, Entail: {:.2f}%".format(
        nli_contradict_class_ratio*100,
        nli_neutral_class_ratio*100,
        nli_entail_class_ratio*100
    ))

    res_path = args.hyp_path + f'_{args.outer_strategy}_factuality_results.txt'
    with open(res_path, 'a') as outfile:
        res_obj = {
            "avg_hallu_ner_ratio": avg_hallu_ner_ratio,
            "nli_contradict_class_ratio": nli_contradict_class_ratio,
            "nli_neutral_class_ratio": nli_neutral_class_ratio, 
            "nli_entail_class_ratio": nli_entail_class_ratio,
        }
        json.dump(res_obj, outfile)
        outfile.write("\n")

    ana_path = args.hyp_path + f'_IR{args.retrieved_num}_{args.outer_strategy}_analysis.txt'
    with open(ana_path, 'w') as outfile:
        json.dump(all_analysis_list, outfile)
        outfile.write("\n")

    # save example NE score 
    ne_path = args.hyp_path + f'_IR{args.retrieved_num}_{args.outer_strategy}_example_NE.txt'
    with open(ne_path, 'w') as outfile:
        for ne in final_hallu_ner_score:
            outfile.write(str(ne) + '\n')

    # save example NLI score
    nli_path = args.hyp_path + f'_IR{args.retrieved_num}_{args.outer_strategy}_example_NLI_entail.txt'
    with open(nli_path, 'w') as outfile:
        for nli in final_entail_prob:
            outfile.write(str(nli) + '\n')

if __name__ == '__main__':
    args = args_parser()
    main(args)
