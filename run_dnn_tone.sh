#!/bin/bash

. ./cmd.sh
. ./path.sh

nj=2
stage=9

# data prepare.
if [ $stage -le 0 ]; then 

	local/normal/matbn_data_prepare.sh data/train data/dev data/test /mnt/corpus/MATBN || exit 1;
	
	local/normal/matbn_dict_prepare.sh local/normal/dict data/local/dict || exit 1;
	
	utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SIL>" data/local/lang data/lang || exit 1;
	
	local/normal/matbn_lm_prepare.sh data/lang data/lang_test || exit 1;
	
fi #stage 0

# extract feature
if [ $stage -le 1 ]; then 

mfccdir=mfcc_pitch1
for x in train test dev; do 
	steps/make_mfcc_pitch.sh --nj $nj --cmd "utils/run.pl" data/$x exp/make_mfcc_pitch1/$x $mfccdir || exit 1;
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc_pitch1/$x $mfccdir || exit 1;
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

# data prepare. (for tone)
if [ $stage -le 3 ]; then
	
	local/tone/matbn_tone_data_prepare.sh data/lang/phones.txt \
								data_tone/train exp/mono0a_train_ali \
								data_tone/dev exp/mono0a_dev_ali \
								data_tone/test exp/mono0a_test_ali \
								/mnt/corpus/MATBN || exit 1;
								
	local/tone/matbn_tone_dict_prepare.sh local/tone/dict data_tone/local/dict || exit 1;
	
	utils/prepare_lang.sh --position-dependent-phones false --num-sil-states 3 --num-nonsil-states 1 \
						data_tone/local/dict "<SIL>" data_tone/local/lang data_tone/lang || exit 1;
	
	local/tone/matbn_tone_lm_prepare.sh data_tone/lang data_tone/lang_test || exit 1;

fi #stage 3

# extract feature (for tone)
if [ $stage -le 4 ]; then

	for x in train test dev; do 
		local/tone/make_mfcc_pitch_lda.sh $x data/$x data_tone/$x mfcc_pitch_lda
		local/tone/make_mfcc_pitch_segments.sh $x data_tone/$x data_tone/$x/segments data_tone/$x mfcc_pitch_segments
		steps/compute_cmvn_stats.sh data_tone/$x exp/make_mfcc_pitch_segments/$x mfcc_pitch_segments || exit 1;
	done
		
fi #stage 4

# training mono phone (for tone)
if [ $stage -le 5 ]; then

	for x in train dev; do 
		steps/train_mono.sh --boost-silence 1.25 --nj $nj \
			data_tone/$x data_tone/lang exp/mono0a_tone_$x || exit 1;
	
		ali_dir="mono0a_tone_"$x"_ali"
		steps/align_si.sh --boost-silence 1.25 --nj $nj  \
			data_tone/$x data_tone/lang exp/mono0a_tone_$x exp/$ali_dir || exit 1;
	done	

fi #stage 5

# pre-train (for tone)
if [ $stage -le 6 ]; then
	#RBM pretrain
	dir=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain
	$cuda_cmd $dir/_pretrain_dbn.log \
		steps/nnet/pretrain_dbn.sh --nn-depth 4 --hid-dim 1024 --rbm-iter 3 data_tone/train $dir

fi #stage 6

# DNN training (for tone)
if [ $stage -le 7 ]; then

	dir=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024
	ali=exp/mono0a_tone_train_ali
	ali_dev=exp/mono0a_tone_dev_ali
	feature_transform=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain/final.feature_transform
	dbn=exp/tri3a_mfcc_pitch_dnn_nd4_hd1024_pretrain/4.dbn
	$cuda_cmd $dir/_train_nnet.log \
		steps/nnet/train.sh --feature-transform $feature_transform \
		--dbn $dbn --hid-layers 0 --learn-rate 0.008 \
		data_tone/train data_tone/dev data_tone/lang $ali $ali_dev $dir || exit 1;

fi #stage 7

# make graph. (for tone)
if [ $stage -le 8 ]; then

	for x in lang_test; do 
		lang_dir=graph_tgpr_$x
		utils/mkgraph.sh --mono data_tone/$x exp/tri3a_mfcc_pitch_dnn_nd4_hd1024 exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_dir
	done

fi #stage 8

# NN decode. (for tone)
if [ $stage -le 9 ]; then

	for x in lang_test; do 
		lang_dir=graph_tgpr_$x
		lang_decode=decode_test_$x
		steps/nnet/decode.sh --nj $nj --acwt 0.10 --config conf/decode_dnn.config \
			exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_dir data_tone/test exp/tri3a_mfcc_pitch_dnn_nd4_hd1024/$lang_decode
	done

fi #stage 9

# display score.
if [ $stage -le 10 ]; then

	for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

fi #stage 10


