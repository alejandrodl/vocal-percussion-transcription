module load gcc/8.2.0
export KALDI_ROOT=/import/linux/kaldi
#[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
[ -f /import/research_c4dm/ed308/kaldi/tools/env.sh ] && . /import/research_c4dm/ed308/kaldi/tools/env.sh
export PATH=$KALDI_ROOT/egs/timit/s5/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

export LC_ALL=C
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.7/lib:/import/linux/intel-mkl/mkl/lib/intel64/
