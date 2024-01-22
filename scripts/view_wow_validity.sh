your_log_name_list=(log1 log2 log3)  # Replace with actual log names

for name in "${your_log_name_list[@]}"; do
do

echo $name
tail -2 log/log-${name}-answer
echo

done