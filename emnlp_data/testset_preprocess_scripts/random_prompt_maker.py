import random
import json
import tqdm

# for wow
# data = []
# with open('train_processed.txt', 'r', encoding='utf-8') as r:
#     data = r.readlines()

# for split in ['seen', 'unseen']:
#     with open(f'random_prompts/{split}_random_prompt.txt', 'w', encoding='utf-8') as w:
#         for i in range(500): # lines, examples
#             random.shuffle(data)
#             cur_prompt = data[:50]
#             w.write(str(cur_prompt).strip() + '\n')
#             # print (cur_prompt)



with open('/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/retriever/nq-dev.json', 'r', encoding='utf-8') as infile:
    data_list = json.load(infile)
    print (data_list[0].keys()) # dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])


train = '/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/retriever/nq-train.json'
# with open(train, 'r', encoding='utf-8') as infile, \
#     open('../nq/random_prompts/nq_test_random_prompt.txt', 'w', encoding='utf-8') as outfile:
#     data_list = json.load(infile)
#     for i in tqdm.tqdm(range(500)):
#         # random.shuffle(data_list)
#         id_list = random.sample(list(range(len(data_list))), 300)
#         cur_prompt = []
#         # for data in data_list:
#         for id in id_list:
#             data = data_list[id]
#             if not data['positive_ctxs'] or not data['answers']:
#                 continue
            
#             query = data['question'].strip()
#             answer = data['answers'][0].strip()

#             knowledge = data['positive_ctxs'][0]['text'].strip()
#             topic = data['positive_ctxs'][0]['title'].strip()

#             if len(knowledge.split(' ')) > 350:
#                 continue
#             cur_prompt.append(f'{topic}\t{query}\t{knowledge}\t{answer}\n')
#             if len(cur_prompt) == 50:
#                 break
#         outfile.write(str(cur_prompt).strip() + '\n')