import json

# for wow
# query_len, knowledge_len = [], []
# with open('train_processed.txt', 'r', encoding='utf-8') as r:
#     data = r.readlines()
#     print ('data size: ', len(data))
#     for i, line in enumerate(data):
#         parts = [e.strip() for e in line.strip().split('\t')]
#         # assert len(parts) == 4, (i, len(parts), parts)
#         if len(parts) != 4:
#             print(i, len(parts), parts)
#             continue
#         topic, history, knowledge, response = parts
#         query = history.split(' [SEP] ')[-1]
#         query_len.append(len(query.split(' ')))
#         knowledge_len.append(len(knowledge.split(' ')))
#     assert len(query_len) == len(knowledge_len), 'length not equal'
#     print('query len: ', sum(query_len) / len(query_len)) # 14.6
#     print('knowledge len: ', sum(knowledge_len) / len(knowledge_len)) # 21.1

# for nq
train = '/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/retriever/nq-train.json'
test = '/misc/kfdata01/kf_grp/lchen/DPR/dpr/downloads/data/gold_passages_info/nq_test.json'
query_len, knowledge_len = [], []
with open(test, 'r') as infile:
    data_list = json.load(infile)['data']
    for data in data_list:
        if not data['context'] or not data['short_answers']:
            continue
        query = data['question']
        knowledge = data['context']
        query_len.append(len(query.split(' ')))
        knowledge_len.append(len(knowledge.split(' ')))
    print('query len: ', sum(query_len) / len(query_len)) # 9.0
    print('knowledge len: ', sum(knowledge_len) / len(knowledge_len)) # 297.2

print (len(knowledge_len)) # 1868
print (max(knowledge_len))
li = [0] * 21
for l in knowledge_len:
    if l > 1000:
        continue
    li[l // 50] += 1
print (li)

# [274, 582, 403, 247, 85, 46, 25, 19, 21, 7, 9, 10, 5, 6, 5, 7, 1, 1, 2, 0, 0]