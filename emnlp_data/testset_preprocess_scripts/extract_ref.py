

def read_testfile(ref_path):
    # testset = []
    with open(ref_path, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            # topic, query, knowledge, response
            assert len(parts) == 4, parts
            # testset.append(parts)
            print (parts[-2])
    # return testset

read_testfile('/misc/kfdata01/kf_grp/lchen/EMNLP23/experiments/emnlp_data/wow/random_testset/seen_random_testset.txt')