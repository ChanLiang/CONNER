
for name in your_prediction

do

ref=emnlp_data/wow/random_testset/seen_random_testset.txt
hyp=${name}/seen_knowledge

exp_name=relevance_${name}
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u relevance.py  \
--hyp_path $hyp \
--ref_path $ref 1>log/log-${exp_name} 2>&1 

echo 

wait
done
