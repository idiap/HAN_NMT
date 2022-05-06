#!/bin/bash
moses_scripts=/content/mosesdecoder/scripts

# zh_segment_home=/idiap/home/lmiculicich/.cache/pip/wheels/ce/32/de/c2be1db5f30804bc7f146ff698c52963f8aa11ba5049811b0d
#kpu_preproc_dir=/fs/zisa0/bhaddow/code/preprocess/build/bin

max_len=200

# export PYTHONPATH=$zh_segment_home

src=de
tgt=en
pair=$src-$tgt


# Tokenise the English part
cat corpus.$tgt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $tgt | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $tgt  \
> corpus.tok.$tgt

#Segment the Chinese part
python -m jieba -d ' ' < corpus.$src > corpus.tok.$src 

#
###
#### Clean
#$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model truecase-model.$tgt -corpus corpus.tok.$tgt
$moses_scripts/recaser/truecase.perl < corpus.tok.$tgt > corpus.tc.$tgt -model truecase-model.$tgt

ln -s corpus.tok.$src  corpus.tc.$src
#
#  
# dev sets
for devset in dev2010 tst2010 tst2011 tst2012 tst2013; do
  for lang  in $src $tgt; do
    if [ $lang = $tgt ]; then
      side="src"
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < IWSLT15.TED.$devset.$src-$tgt.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model truecase-model.$lang \
      > IWSLT15.TED.$devset.tc.$lang
    else
      side="ref"
      python -m jieba -d ' '  < IWSLT15.TED.$devset.$src-$tgt.$lang > IWSLT15.TED.$devset.tc.$lang
    fi
    
  done

done

# DATA_DIR=/content/gdrive/MyDrive/2015-01/texts/zh/en/zh-en
DATA_DIR=$1
echo "data_dir: $DATA_DIR"
python /content/HAN_NMT/full_source/preprocess.py -train_src $DATA_DIR/corpus.tc.$src -train_tgt $DATA_DIR/corpus.tc.$tgt -train_doc $DATA_DIR/corpus.doc -valid_src $DATA_DIR/IWSLT15.TED.dev2010.tc.$src -valid_tgt $DATA_DIR/IWSLT15.TED.dev2010.tc.$tgt -valid_doc $DATA_DIR/IWSLT15.TED.dev2010.$src-$tgt.doc -save_data $DATA_DIR/IWSLT15.TED -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80

