'''
pip install sgnlp
pip uninstall nvidia_cublas_cu11
'''
import numpy as np
from tqdm import tqdm
import json
from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor

# Load Model
config = CoherenceMomentumConfig.from_pretrained(
    "coherence-momentum"
)
model = CoherenceMomentumModel.from_pretrained(
    "coherence-momentum",
    config=config
)

model.cuda()

preprocessor = CoherenceMomentumPreprocessor(config.model_size, config.max_len)

# Example text inputs
text1 = "Companies listed below reported quarterly profit substantially different from the average of analysts ' " \
        "estimates . The companies are followed by at least three analysts , and had a minimum five-cent change in " \
        "actual earnings per share . Estimated and actual results involving losses are omitted . The percent " \
        "difference compares actual profit with the 30-day estimate where at least three analysts have issues " \
        "forecasts in the past 30 days . Otherwise , actual profit is compared with the 300-day estimate . " \
        "Source : Zacks Investment Research"
text2 = "The companies are followed by at least three analysts , and had a minimum five-cent change in actual " \
        "earnings per share . The percent difference compares actual profit with the 30-day estimate where at least " \
        "three analysts have issues forecasts in the past 30 days . Otherwise , actual profit is compared with the " \
        "300-day estimate . Source : Zacks Investment Research. Companies listed below reported quarterly profit " \
        "substantially different from the average of analysts ' estimates . Estimated and actual results involving " \
        "losses are omitted ."


def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_path", type=str, default='./emnlp_data/nq/random_testset/nq_ref')
    args = parser.parse_args()
    return args

def calculate_coherence(sentences):
    print('calculate coherence scores...')
    scores = []
    for s_list in tqdm(sentences):
        inputs = preprocessor([s_list])
        score = model.get_main_score(inputs["tokenized_texts"].cuda()).item()
        scores.append(score)
    return scores

def read_hyp(hyp_path):
    hyps = []
    with open(hyp_path, 'r') as infile:
        for line in infile:
            hyps.append(line.strip())
    return hyps


if __name__ == '__main__':
    args = args_parser()
    hyps = read_hyp(args.hyp_path)
    assert len(hyps) == 500, len(hyps)

    scores = calculate_coherence(hyps)
    assert len(scores) == 500, len(scores)

    with open(args.hyp_path + '_avg_coh_para', 'w') as outfile:
        json.dump(scores, outfile)
        outfile.write('\n')
        outfile.write(f'{max(scores)}\t{min(scores)}')
