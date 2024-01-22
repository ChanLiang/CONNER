# for model in flan-t5-xl flan-t5-large flan-t5-base flan-t5-small
for model in flan-t5-11B
do
split=seen
data=few-shot
head -3865 /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge > /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_

# seen + zero-shot
split=seen
data=zero-shot
head -3865 /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter > /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter_


# unseen + few-shot
split=unseen
data=few-shot
head -3924 /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge > /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_


# unseen + zero-shot
split=unseen
data=zero-shot
head -3924 /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter > /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter_

done