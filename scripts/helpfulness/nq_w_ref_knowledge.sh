export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

task=nq
exp_name=nq_llama_65B_w_ref_knowledge

# debug=True
debug=False
testfile=../emnlp23/emnlp_data/nq/random_testset/nq_test_random_testset.txt
promptfile=../emnlp23/emnlp_data/nq/random_prompts/nq_test_random_prompt.txt

downstream_model=llama-65B
knowledge_type=w_ref_knowledge
zero_shot=False

export TRANSFORMERS_CACHE='YOUR_DIR'
export HF_HOME='YOUR_DIR'
export HUGGINGFACE_HUB_CACHE='YOUR_DIR'

# export CUDA_VISIBLE_DEVICES=1,2,3
python3 -u helpfulness.py \
--exp_name $exp_name \
--task $task \
--zero_shot $zero_shot \
--debug $debug \
--testfile $testfile \
--promptfile $promptfile \
--downstream_model $downstream_model \
--knowledge_type $knowledge_type 1>log/$exp_name.log 2>&1
