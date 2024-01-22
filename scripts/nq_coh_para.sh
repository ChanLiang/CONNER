# env: base

for name in your_prediction_dir
do

hyp=${name}/nq_hyp

exp_name=discourse_coherence_${name}
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u discourse-coherence.py  \
--hyp_path $hyp 1>log/log-${exp_name} 2>&1 

echo 

wait
done
