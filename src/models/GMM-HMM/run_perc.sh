#!/bin/bash

set -e # exit on error

. ./path.sh
. ./cmd.sh

# This script checks if external tools required to build the
# pronunciation and language models are installed properly.

#./local/check_tools.sh || exit 1


dataset_path=$1

echo "Using steps and utils from WSJ recipe"
[[ ! -L "wav" ]] && ln -s $dataset_path wav
[[ ! -L "steps" ]] && ln -s $KALDI_ROOT/egs/wsj/s5/steps
[[ ! -L "utils" ]] && ln -s $KALDI_ROOT/egs/wsj/s5/utils
[[ ! -L "rnnlm" ]] && ln -s $KALDI_ROOT/egs/wsj/s5/rnnlm


# Begin configuration section

nj=5
stage=1


trainset=train
testset=test

# End configuration section
. ./utils/parse_options.sh


affix=

mfccdir=mfcc

echo; echo "===== Starting at  $(date +"%D_%T") ====="; echo


if [ $stage -le 1 ]; then

    ### STAGE 1: DICTIONARY CREATION
    ### We begin with creating a dictionary WFST and relevant files
    ### for mapping instrument types to phoneme sequences (or pronunciations)
    ### Here, we take on the automatic speech recognition approach,
    ### and treat instrument types as 'words'.
    ### This corresponds to the 'pronunciation model' in GMM-HMM
    ### speech recognition.
    mkdir -p data/local/dict${affix}
    cp conf/corpus.txt data/local/corpus.txt  # Corpus.txt for language model
    local/prepare_dict.sh --words 4 --affix ${affix} --corpus_dir data/local/corpus.txt  ### The vocabulary size is predetermined based on dataset properties.
    utils/prepare_lang.sh --share-silence-phones true --position_dependent_phones true \
        data/local/dict${affix} "<UNK>" data/local/lang${affix} data/lang${affix}

fi

if [ $stage -le 2 ]; then
    ### STAGE 2: LANGUAGE MODEL
    ### We train a language model on the instrument-type transcriptions
    ### of the training data. Since size of the training data is small,
    ### we employ the statistical n-gram approximation for building the LM
    ### Here, we build the LM on 3 and 4 - grams.

    local/train_lms_srilm.sh \
        --train-text data/local/corpus_perc.txt \
        --oov-symbol "<UNK>" --words-file data/lang$affix/words.txt \
        data/ data/srilm$affix
        

    # Compiles G for DSing trigram LM
    utils/format_lm.sh  data/lang$affix data/srilm$affix/best_3gram.gz data/local/dict${affix}/lexicon.txt data/lang_3G$affix
    utils/format_lm.sh  data/lang$affix data/srilm$affix/best_4gram.gz data/local/dict${affix}/lexicon.txt data/lang_4G$affix

fi


if [[ $stage -le 3 ]]; then
  ### STAGE 3: FEATURE EXTRACTION
  ### We extract 13-band MFCC features at this stage.
  echo
  echo "============================="
  echo "---- MFCC FEATURES EXTRACTION ----"
  echo "=====  $(date +"%D_%T") ====="

  for datadir in  $testset ; do #$trainset
    echo; echo "---- ${datadir}"
    utils/fix_data_dir.sh data/${datadir}
    steps/make_mfcc.sh --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
       --nj 5 data/${datadir} exp$affix/make_mfcc/${datadir} $mfccdir
    steps/compute_cmvn_stats.sh data/${datadir}
    utils/fix_data_dir.sh data/${datadir}
  done

fi

if [ $stage -le 4 ]; then
  ### STAGE 4: DATA AUGMENTATION
  echo " ====================== "
  echo " ---  Data Augmentation - 3-way Speed Perturbation  ---  "
  echo " ====================== "
  echo 
  echo "$0: preparing directory for speed-perturbed data"
  utils/data/perturb_data_dir_speed_3way.sh data/${trainset} data/${trainset}_sp
fi


if [[ $stage -le 5 ]]; then
  ### STAGE 5: FEATURE EXTRACTION FOR THE AUGMENTED DATA
  echo
  echo "============================="
  echo "---- MFCC FEATURES EXTRACTION ----"
  echo "=====  $(date +"%D_%T") ====="

  for datadir in ${trainset}_sp ; do
    echo; echo "---- ${datadir}"
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires    
    utils/fix_data_dir.sh data/${datadir}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf \
       --nj 9 data/${datadir}_hires exp$affix/make_mfcc/${datadir}_hires $mfccdir
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done

fi



if [[ $stage -le 6 ]]; then
  ### STAGE 6: TRAINING GMM-HMM ---- MONOPHONE MODEL
    echo
    echo "============================="
    echo "-------- Train GMM ----------"
    echo
    echo "Monophone"
    echo "=====  $(date +"%D_%T") ====="

    steps/train_mono.sh --nj $nj --cmd "$train_cmd" --totgauss 500 --boost-silence 1.25 \
      --num_iters 40 data/${trainset} data/lang$affix exp$affix/mono

    steps/align_si.sh --nj $nj --cmd "$train_cmd" --beam 40 --retry_beam 200 \
      data/${trainset} data/lang$affix exp$affix/mono exp$affix/mono_ali

    utils/mkgraph.sh data/lang_4G$affix exp$affix/mono exp$affix/mono/graph

    # TESTING STAGE - MONOPHONE MODEL

    echo; echo "--------decode test"; echo
    steps/decode.sh --config conf/decode.config --nj 5 --cmd "$decode_cmd" \
      --scoring-opts "--min-lmwt 10 --max-lmwt 20" --num-threads 4 --beam 50 \
      exp$affix/mono/graph data/$testset exp$affix/mono/decode_test

fi


if [[ $stage -le 7 ]];then
    ### STAGE 7: TRAINING GMM-HMM - TRIPHONE MODEL (DELTA + DELTA-DELTA)
    echo
    echo "Tri 1 - delta-based triphones"
    echo "=====  $(date +"%D_%T") ====="

    steps/train_deltas.sh  --cmd "$train_cmd" --boost_silence 1.25 --beam 30 --retry_beam 100 300 1500 \
      data/${trainset} data/lang$affix exp$affix/mono_ali exp$affix/tri1

    steps/align_si.sh --nj $nj --cmd "$train_cmd" --beam 30 --retry_beam 100  \
     data/${trainset} data/lang$affix exp$affix/tri1 exp$affix/tri1_ali

    utils/mkgraph.sh data/lang_4G$affix exp$affix/tri1 exp$affix/tri1/graph
    
    # TESTING STAGE  
    steps/decode.sh --config conf/decode.config --nj 5 --cmd "$decode_cmd" \
      --scoring-opts "--min-lmwt 10 --max-lmwt 20" --num-threads 4 --beam 30 \
      exp$affix/tri1/graph data/$testset exp$affix/tri1/decode_test
fi



### STAGE 8 & 9: TRAINING GMM-HMM - TRIPHONE MODEL ON SPEAKER ADAPTIVE FEATURES

if [[ $stage -le 8 ]];then

    echo
    echo "Tri 2 - LDA-MLLT triphones"
    echo "=====  $(date +"%D_%T") ====="

    steps/train_lda_mllt.sh --cmd "$train_cmd" --beam 40 --retry_beam 80 5000 40000 \
      data/${trainset} data/lang$affix exp$affix/tri1_ali exp$affix/tri2b

    steps/align_si.sh --nj $nj --cmd "$train_cmd" --beam 40 --retry_beam 100  \
      data/${trainset} data/lang$affix exp$affix/tri2b exp$affix/tri2b_ali

fi

if [[ $stage -le 9 ]];then

    echo
    echo "Tri 3 - SAT triphones"
    echo "=====  $(date +"%D_%T") ====="
   
    steps/train_sat.sh --cmd "$train_cmd" --beam 40 --retry_beam 100 6000 70000 \
      data/${trainset} data/lang$affix exp$affix/tri2b_ali exp$affix/tri3b

    utils/mkgraph.sh data/lang_4G$affix exp$affix/tri3b exp$affix/tri3b/graph
   
    echo
    echo "------ End Train GMM --------"
    echo "=====  $(date +"%D_%T") ====="
fi

if [[ $stage -le 10 ]]; then
    ### FINAL TESTING STAGE
    echo
    echo "============================="
    echo "------- Decode TRI3B --------"
    echo "=====  $(date +"%D_%T") ====="
    echo
    for datadir in $testset; do
      steps/decode_fmllr.sh --config conf/decode.config --nj 5 --cmd "$decode_cmd" \
        --scoring-opts "--min-lmwt 10 --max-lmwt 20" --num-threads 4 --beam 30 \
        exp$affix/tri3b/graph data/${datadir} exp$affix/tri3b/decode_${datadir}

    done
fi


echo
echo "=====  $(date +"%D_%T") ====="
echo "===== PROCESS ENDED ====="
echo

exit 1
