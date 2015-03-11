#!/bin/bash

. ./cmd.sh
. ./path.sh

nj=1
stage=7

# data prepare.
if [ $stage -le 0 ]; then 

	local/normal/matbn_data_prepare.sh data/train data/dev data/test /mnt/corpus/MATBN;
	
	local/normal/matbn_dict_prepare.sh local/normal/dict data/local/dict;
	
	utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SIL>" data/local/lang data/lang;
	
	local/normal/format_local_lms.sh data/lang local/normal/lm/srilm.o3g.kn data/lang_test
	
fi #stage 0

# extract feature
if [ $stage -le 1 ]; then 

mfccdir=mfcc_pitch
for x in train test dev; do 
	steps/make_mfcc_pitch.sh --nj $nj --cmd "utils/run.pl" data/$x exp/make_mfcc_pitch/$x $mfccdir || exit 1;
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc_pitch/$x $mfccdir || exit 1;
done

fi #stage 1

# training mono phone
if [ $stage -le 2 ]; then

for x in train test dev; do 
	steps/train_mono.sh --boost-silence 1.25 --nj $nj  \
	data/$x data/lang exp/mono0a_$x || exit 1;
	
	ali_dir="mono0a_"$x"_ali"
	steps/align_si.sh --boost-silence 1.25 --nj $nj  \
	data/$x data/lang exp/mono0a_$x exp/$ali_dir || exit 1;
done

fi #stage 2

# training tri phone and LDA
if [ $stage -le 3 ]; then

steps/train_deltas.sh --boost-silence 1.25 --cmd $train_cmd \
    2000 10000 data/train data/lang exp/mono0a_train_ali exp/tri1_train || exit 1;
	
steps/align_si.sh --nj $nj \
  data/train data/lang exp/tri1_train exp/tri1_train_ali || exit 1;
  
steps/train_lda_mllt.sh \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/train data/lang exp/tri1_train_ali exp/tri2b_train || exit 1;
   
steps/align_si.sh  --nj $nj \
  --use-graphs true data/train data/lang exp/tri2b_train exp/tri2b_train_ali  || exit 1;
   
steps/align_si.sh  --nj $nj \
  data/dev data/lang exp/tri2b_train exp/tri2b_dev_ali || exit 1;
  
fi #stage 3

# pre-train (for tone)
if [ $stage -le 4 ]; then
	#RBM pretrain
	dir=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain
	$cuda_cmd $dir/_pretrain_dbn.log \
		steps/nnet/pretrain_dbn.sh --nn-depth 4 --hid-dim 1024 --rbm-iter 3 data/train $dir

fi #stage 4

# DNN training (for tone)
if [ $stage -le 5 ]; then

	dir=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024
	ali=exp/tri2b_train_ali
	ali_dev=exp/tri2b_dev_ali
	feature_transform=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain/final.feature_transform
	dbn=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain/4.dbn
	$cuda_cmd $dir/_train_nnet.log \
		steps/nnet/train.sh --feature-transform $feature_transform \
		--dbn $dbn --hid-layers 0 --learn-rate 0.008 \
		data/train data/dev data/lang $ali $ali_dev $dir || exit 1;

fi #stage 5

# make graph. (for tone)
if [ $stage -le 6 ]; then

	for x in lang_test; do 
		lang_dir=graph_tgpr_$x
		utils/mkgraph.sh data/$x exp/tri3a_mfcc_pitch_dnn_nd4_hd1024 exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_dir
	done

fi #stage 6

# NN decode. (for tone)
if [ $stage -le 7 ]; then

	for x in lang_test; do 
		lang_dir=graph_tgpr_$x
		lang_decode=decode_test_$x
		steps/nnet/decode.sh --nj $nj --acwt 0.10 --config conf/decode_dnn.config \
			exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_dir data/test exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_decode
	done

fi #stage 7

# display score.
if [ $stage -le 8 ]; then

	for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

fi #stage 8


