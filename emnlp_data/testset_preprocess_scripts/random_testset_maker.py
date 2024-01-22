import random
import json

# dataset = 'wow'
# for split in ['seen', 'unseen']:
#     data = []
#     with open(f'../{dataset}/test{split}_processed.txt', 'r', encoding='utf-8') as r:
#         data = r.readlines()
#     with open(f'random_testset/{split}_random_testset.txt', 'w', encoding='utf-8') as w:
#         random.shuffle(data)
#         testset = data[:500]
#         w.writelines(testset)



train = '/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/retriever/nq-train.json'
test = '/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/gold_passages_info/nq_test.json'
cnt = 0
with open(test, 'r') as infile, \
    open('../nq/random_testset/nq_test_random_testset.txt', 'w') as outfile:
    data_list = json.load(infile)['data']
    for data in data_list:
        if not data['context'] or not data['short_answers']:
            continue
        
        p = random.random()
        if p > 0.35:
            continue
        topic = data['title'].strip()
        query = data['question'].strip()
        knowledge = data['context'].strip()
        answer = data['short_answers'][0].strip()
        outfile.write(f'{topic}\t{query}\t{knowledge}\t{answer}\n')
        cnt += 1
        if cnt == 500:
            break
        