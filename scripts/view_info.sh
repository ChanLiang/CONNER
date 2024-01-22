for name in your_dir
do

echo $name
exp_name=info_${name}
tail -1 log/log-${exp_name}
echo ' '

done