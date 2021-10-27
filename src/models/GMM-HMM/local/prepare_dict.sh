#!/bin/bash

#adapted from ami and chime5 dict preparation script


# Begin configuration section.
words=5000
# End configuration section
affix=

corpus_dir=data/local/corpus.txt
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

mkdir -p data



dir=data/local/dict$affix
mkdir -p $dir

cp conf/dict$affix/* $dir/
#exit 1
echo "$0: Preparing files in $dir"
# Silence phones
#for w in SIL SPN; do echo $w; done > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt



# Add prons for laughter, noise, oov
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon_raw.txt > $dir/lexicon2_raw.txt || exit 1;


# we keep all words from the cmudict in the lexicon
# might reduce OOV rate on dev and test
cat $dir/lexicon_raw.txt  \
   <( echo "<UNK> SPN" >> $dir/lexicon.txt \
      echo "<silence> SIL" >> $dir/lexicon.txt \
    )  |  sort -u > $dir/lexicon.txt # tr a-z A-Z  |


cat $corpus_dir  | \
  awk '{for (n=1;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
  sort -nr > $dir/word_counts_b





echo "<UNK> SPN" >> $dir/lexicon.txt
echo "<silence> SIL" >> $dir/lexicon.txt


sed -e 's/ / 1.0\t/' data/local/dict$affix/lexicon.txt > data/local/dict$affix/lexiconp.txt

utils/validate_dict_dir.pl $dir
exit 0;
