import torch
import tqdm
import argparse
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import DataLoader, TensorDataset
import json

# env: conda activate D3
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # model.half()
    model.eval()
    return tokenizer, model, device

def read_testfile(ref_path):
    testset = []
    with open(ref_path, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            # topic, query, knowledge, response
            assert len(parts) == 4, parts
            testset.append(parts)
    return testset

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps

def get_dataloader(input_examples, tokenizer, device, batch_size=256):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1'],
        max_length=512,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, token_type_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def load_data(ref_path, hyp_path, tokenizer, device, batch_size=256):
    testset = read_testfile(ref_path)
    hyps = read_hyp(hyp_path)
    assert len(testset) == len(hyps), (len(testset), len(hyps))
    # examples = [InputExample(str(i), testset[i][1], hyps[i], '0') for i in range(len(testset))]
    examples = [InputExample(str(i), testset[i][1], hyps[i], '0') for i in range(len(testset)) if hyps[i].strip()]
    test_dataloader = get_dataloader(examples, tokenizer, device, batch_size=batch_size)
    return test_dataloader, examples

def batch_inference(model, dataloader):
    all_logits = None
    with torch.no_grad():
        # for batch in tqdm.tqdm(dataloader):
        for batch in dataloader:
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2]}
            outputs = model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].cpu().detach()
            else: # [n, 2], 每个batch直接cat到第一个维度上 
                all_logits = torch.cat((all_logits, outputs[0].cpu().detach()), dim=0)
    results = torch.argmax(all_logits, dim=1) # [n]
    probs = torch.nn.functional.softmax(all_logits, dim=-1)
    # return results, probs[torch.arange(probs.size(0)), results]
    return results, probs[:, 1]

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default='/misc/kfdata01/kf_grp/lchen/EMNLP23/experiments/emnlp_data/nq/random_testset/nq_test_random_testset.txt')
    parser.add_argument("--hyp_path", type=str, default='/misc/kfdata01/kf_grp/lchen/EMNLP23/experiments/emnlp_data/nq/random_testset/nq_ref')
    parser.add_argument("--model_path", type=str, default='/misc/kfdata01/kf_grp/lchen/cache/monobert-large-msmarco')
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_parser()

    # 1. load model
    tokenizer, model, device = load_model(args.model_path)
    
    # 2. load data
    test_dataloader, examples = load_data(args.ref_path, args.hyp_path, tokenizer, device, batch_size=args.batch_size)

    # 3. inference
    results, probs = batch_inference(model, test_dataloader)
    # print (results, probs)
    # probs = torch.tensor([p for p in probs if p > 0.01])
    print (torch.sum(results), torch.mean(probs))

    with open(args.hyp_path + '_rel', 'w') as w:
        # w.write(json.dumps())
        for prob in probs:
            w.write(str(prob.item()) + '\n')

    # 4. print some examples
    # res = results.cpu().tolist()
    # for i in range(500):
    #     idx = res[i]
    #     if idx == 0:
    #         print ('=='*20)
    #         print (examples[i].text_a + '\n')
    #         print (examples[i].text_b)

